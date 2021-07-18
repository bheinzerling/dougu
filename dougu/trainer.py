import sys
import os
import subprocess
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver


from dougu import (
    Configurable,
    WithLog,
    mkdir,
    next_rundir,
    dump_args,
    json_load,
    json_dump,
    )
from dougu.decorators import cached_property
from dougu.torchutil import (
    set_random_seed,
    get_optim,
    get_lr_scheduler,
    log_param_count,
    fix_dataparallel_statedict,
    )

from dougu.experiment_logger import ExperimentLogger


class TrainerBase(Configurable, WithLog):
    args = [
        ('command', dict(type=str)),
        ('--device', dict(type=str, default='cuda:0')),
        ('--random-seed', dict(type=int, default=2)),
        ('--runid', dict(type=str)),
        ('--batch-size', dict(type=int, default=128)),
        ('--eval-batch-size', dict(type=int, default=128)),
        ('--eval-every', dict(type=int, default=1)),
        ('--first-eval-epoch', dict(type=int, default=1)),
        ('--optim', dict(type=str, default='adam')),
        ('--lr', dict(type=float, default=0.001)),
        ('--lr-scheduler', dict(type=str, default='plateau')),
        ('--lr-scheduler-patience', dict(type=int, default=10)),
        ('--lr-metric-name', dict(type=str, default='loss')),
        ('--lr-metric-optimum', dict(type=str, default='min')),
        ('--momentum', dict(type=float, default=0.0)),
        ('--weight-decay', dict(type=float, default=0.0)),
        ('--adam-betas', dict(type=float, nargs='+', default=[0.9, 0.999])),
        ('--adam-eps', dict(type=float, default=1e-8)),
        ('--warmup-steps', dict(type=int, default=1000)),
        ('--n-train-steps', dict(type=int, default=100000)),
        ('--early-stopping', dict(type=int, default=10)),
        ('--early-stopping-burnin', dict(type=int, default=20)),
        ('--max-epochs', dict(type=int, default=1000)),
        ('--no-checkpoints', dict(action='store_true')),
        ('--checkpoint-metric-name', dict(type=str, default='dev_acc')),
        ('--checkpoint-metric-optimum', dict(type=str, default='max')),
        ('--no-fp16', dict(action='store_true')),
        ('--exp-name', dict(type=str, default='dev')),
        ('--outdir', dict(type=Path, default='out')),
        ('--gradient-accumulation-steps', dict(type=int, default=1)),
        ('--max-grad-norm', dict(type=float, default=1.0)),
        ('--dist-backend', dict(type=str, default='nccl')),
        ('--dist-master-addr', dict(type=str, default='127.0.0.1')),
        ('--dist-master-port', dict(type=str, default='29505')),
        ("--dist-init-method", dict(type=str, default="env://")),
        ('--local-rank', dict(type=int, default=0)),
        ('--no-autoscale-lr', dict(action='store_true')),
        ]

    def __init__(self, conf):
        super().__init__(conf)
        self.setup()

    def setup(self):
        self.device = self.conf.device
        set_random_seed(self.conf.random_seed)
        self.misc_init()
        self.setup_distributed()
        if self.is_dist_main():
            self.log(" ".join(sys.argv))
        self.setup_rundir()

        if not self.is_dist_main():
            dist.barrier()
        else:
            self.log('loading data')
            self.data = self.load_data()
            self.data.log_size()
            if self.conf.distributed:
                dist.barrier()
        if not self.is_dist_main():
            self.log('loading data')
            self.data = self.load_data()
        self.setup_model()
        self.setup_optim()
        if self.data.has_train_data():
            self.setup_lr_scheduler()
        self.distribute_model()
        if self.is_dist_main():
            self.log_jobid()
            if self.requires_exp_log:
                self.exp_logger = ExperimentLogger(self.conf, self.exp_params)
            self.setup_bookkeeping()
            if self.data.has_train_data():
                self.setup_early_stopping()
                for event, handler in self.event_handlers_train:
                    self.train_engine.add_event_handler(event, handler)

    @property
    def requires_exp_log(self):
        return self.conf.command != 'cache_dataset'

    def misc_init(self):
        pass

    def log_jobid(self):
        jobid = getattr(self.conf, 'jobid', 0)
        if jobid and self.conf.local_rank == 0:
            jobid_file = self.conf.rundir / 'last_jobid'
            with jobid_file.open('w') as out:
                out.write(str(jobid))

    def setup_rundir(self):
        c = self.conf
        if self.is_dist_main():
            if c.runid is not None:
                c.rundir = mkdir(c.outdir / c.runid)
            else:
                c.rundir = next_rundir()
                c.runid = c.rundir.name
            c.trainer_state_file = c.rundir / 'trainer_state.json'
            c.checkpointer_state_file = c.rundir / 'checkpointer_state.pt'
            dump_args(c, c.rundir / 'conf.json')
            self.log(f'rundir: {c.rundir}')

    def setup_distributed(self):
        self.conf.distributed = 'LOCAL_RANK' in os.environ
        self.log('setting up distributed')
        if self.conf.distributed:
            self.conf.local_rank = int(os.environ['LOCAL_RANK'])
            os.environ['MASTER_ADDR'] = self.conf.dist_master_addr
            os.environ['MASTER_PORT'] = self.conf.dist_master_port
            world_size = torch.cuda.device_count()
            self.log(f'init process group with world size {world_size}')
            dist.init_process_group(
                self.conf.dist_backend,
                init_method=self.conf.dist_init_method,
                rank=self.conf.local_rank,
                world_size=world_size)
            os.environ['RANK'] = str(dist.get_rank())
            self.device = self.conf.local_rank
            self.log(
                f'local rank: {self.conf.local_rank} | '
                f'CUDA device: {self.device}')
            torch.cuda.set_device(self.device)
            if not self.conf.no_autoscale_lr:
                if hasattr(self.conf, 'unscaled_lr'):
                    self.conf.lr = self.conf.unscaled_lr
                self.conf.unscaled_lr = self.conf.lr
                self.conf.lr *= min(world_size, 8)
        else:
            self.conf.local_rank = 0

    def setup_model(self):
        self.model = self.make_model()
        self.maybe_load_model()
        self.model = self.model.to(self.device)
        self.log_model_params()

    def is_dist_main(self):
        return getattr(self.conf, 'local_rank', 0) == 0

    def distribute_model(self):
        if self.conf.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                output_device=self.conf.local_rank,
                find_unused_parameters=True)

    def setup_early_stopping(self):
        if self.conf.early_stopping > 0:
            from dougu.igniteutil import EarlyStoppingWithBurnin
            handler = EarlyStoppingWithBurnin(
                patience=self.conf.early_stopping,
                score_function=self.checkpoint_score_function,
                trainer=self.train_engine,
                min_delta=1e-6,
                burnin=self.conf.early_stopping_burnin)
            self.eval_engine.add_event_handler(Events.COMPLETED, handler)
            self.log(f'Early stopping patience: {self.conf.early_stopping}')

    @property
    def checkpoint_score_function(self):
        sign = {'max': 1, 'min': -1}[self.checkpoint_metric_optimum]

        def score_function(engine):
            metric_name = self.checkpoint_metric_name
            return sign * self.eval_engine.state.metrics[metric_name]
        return score_function

    @property
    def checkpoint_metric_name(self):
        return self.conf.checkpoint_metric_name

    @property
    def checkpoint_metric_optimum(self):
        return self.conf.checkpoint_metric_optimum

    def make_model(self):
        raise NotImplementedError()

    @property
    def model_file(self):
        return self.conf.model_file

    def maybe_load_model(self):
        if self.model_file:
            self.log(f'loading model {self.model_file}')
            state_dict = torch.load(self.model_file, map_location='cpu')
            if self.model_file.name.startswith('checkpoint'):
                state_dict = state_dict['model']
                state_dict = fix_dataparallel_statedict(self.model, state_dict)
            self.model.load_state_dict(state_dict)

    @property
    def last_checkpoint_file(self):
        checkpoints = list(self.conf.rundir.glob('checkpoint_*.pt'))
        if checkpoints:
            last = sorted(checkpoints, key=os.path.getmtime)[-1]
            self.log(f'Found checkpoint: {last}')
            return last
        else:
            self.log('No checkpoint found.')
            return None

    def log_model_params(self):
        if self.is_dist_main():
            log_param_count(self.model, self.log)

    def setup_optim(self):
        self.optim = self.make_optim()

    def setup_lr_scheduler(self):
        self.lr_scheduler = get_lr_scheduler(
            self.conf,
            self.optim,
            optimum=self.conf.lr_metric_optimum,
            n_train_steps=self.n_train_steps)

    def make_optim(self):
        return get_optim(
            self.conf,
            self.model,
            additional_params_dict=self.additional_params_dict)

    @property
    def additional_params_dict(self):
        return None

    def load_data(self):
        raise NotImplementedError()

    @property
    def n_train_steps(self):
        return len(self.data.train_loader) * self.conf.max_epochs

    @cached_property
    def train_metrics(self):
        return self.metrics(prefix='train')

    @cached_property
    def dev_metrics(self):
        return self.metrics(prefix='dev')

    @cached_property
    def test_metrics(self):
        return self.metrics(prefix='test')

    @property
    def eval_event(self):
        def event_filter(engine, event):
            if event % self.conf.eval_every != 0:
                return False
            return event >= self.conf.first_eval_epoch

        return Events.EPOCH_COMPLETED(event_filter=event_filter)

    @cached_property
    def train_engine(self):
        engine = Engine(self.make_train_step())
        if not self.is_dist_main():
            engine.logger.disabled = True
        for metric_name, metric in self.train_metrics.items():
            metric.attach(engine, metric_name)

        if not hasattr(self, 'lr_scheduler') or self.conf.lr_scheduler == 'plateau':
            self.lr_scheduler_train_step = lambda: None
        else:
            if self.conf.lr_scheduler == 'warmup_linear':
                if self.is_dist_main():
                    self.log(f'n_train_steps: {self.n_train_steps}')
            self.lr_scheduler_train_step = self.lr_scheduler.step

        @engine.on(Events.EPOCH_STARTED)
        def set_train_sampler_epoch(_):
            sampler = self.data.train_loader.sampler
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(engine.state.epoch)

        if self.is_dist_main():
            @engine.on(Events.EPOCH_COMPLETED)
            def log_train_metrics(_):
                self.log_metrics(self.train_engine.state.metrics)

            @engine.on(self.eval_event)
            def run_eval(_):
                self.log_results('train', engine.state.metrics)
                self.eval_engine.run(self.data.dev_loader)
                self.log_results('dev', self.eval_engine.state.metrics)
                self.log_metrics(self.eval_engine.state.metrics)

            if not self.conf.no_checkpoints:
                self.eval_engine.add_event_handler(
                    Events.COMPLETED, self.checkpointer, self.checkpoint_dict)
            self.eval_engine.add_event_handler(
                Events.COMPLETED, self.log_checkpoint)

            if self.conf.lr_scheduler == 'plateau':
                def plateau_step(_):
                    metrics = self.eval_engine.state.metrics
                    score = metrics[self.checkpoint_metric_name]
                    return self.lr_scheduler.step(score)

                self.eval_engine.add_event_handler(
                    Events.COMPLETED, plateau_step)

            if hasattr(self.data, 'test_loader'):
                @engine.on(Events.COMPLETED)
                def run_test(_):
                    self.test_engine.run(self.data.test_loader)
                    self.log_results('test', self.test_engine.state.metrics)

        return engine

    @cached_property
    def eval_engine(self):
        engine = Engine(self.make_eval_step())
        for metric_name, metric in self.dev_metrics.items():
            metric.attach(engine, metric_name)
        return engine

    @cached_property
    def test_engine(self):
        engine = Engine(self.make_eval_step())
        for metric_name, metric in self.test_metrics.items():
            metric.attach(engine, metric_name)
        return engine

    def log_results(self, eval_name, metrics):
        metrics_str = ' | '.join(
            f'{metric} {val:.4f}' for metric, val in metrics.items())
        self.log(f"epoch {self.epoch:04d} {eval_name} | {metrics_str}")

    @property
    def actual_model(self):
        return getattr(self.model, 'module', self.model)

    @cached_property
    def checkpointer(self):
        return Checkpoint(
            to_save=self.checkpoint_dict,
            save_handler=DiskSaver(self.conf.rundir, require_empty=False),
            filename_prefix=self.checkpoint_prefix,
            score_name=self.checkpoint_metric_name,
            score_function=self.checkpoint_score_function,
            n_saved=3,
            )

    @property
    def checkpoint_prefix(self):
        return ''

    @property
    def checkpoint_dict(self):
        to_save = {
            'model': self.model,
            'optim': self.optim}
        if getattr(self, 'lr_scheduler', None):
            to_save['lr_scheduler'] = self.lr_scheduler
        if getattr(self, 'amp', None):
            to_save['amp'] = self.amp
        return to_save

    def log_checkpoint(self, engine):
        self.log(f'{self.conf.rundir}/{self.checkpointer.last_checkpoint}')

    @property
    def best_checkpoint(self):
        if not self.checkpointer._saved:
            return None
        return self.conf.rundir / self.checkpointer._saved[-1][1]

    def make_train_step(self):
        accum_steps = self.conf.gradient_accumulation_steps
        n_epoch_steps = len(self.data.train_loader)

        scaler = torch.cuda.amp.GradScaler()

        def train_step(train_engine, batch):
            step = train_engine.state.iteration
            self.model.train()
            batch = {k: v.to(device=self.device) for k, v in batch.items()}
            with autocast(enabled=not self.conf.no_fp16):
                result = self.model(batch)
                loss = result['loss']
            if accum_steps > 1:
                loss /= accum_steps

            if step % accum_steps == 0 or step + 1 == n_epoch_steps:
                if self.conf.max_grad_norm > 0:
                    clip_grad_norm_(
                        self.model.parameters(), self.conf.max_grad_norm)
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                self.lr_scheduler_train_step()
                for p in self.model.parameters():
                    p.grad = None
                scaler.update()
            result['loss'] = loss.item() * accum_steps
            return result
        return train_step

    def make_eval_step(self):
        raise NotImplementedError()

    @property
    def epoch(self):
        return self.train_engine.state.epoch

    @property
    def exp_params(self):
        raise NotImplementedError

    def log_metrics(self, metrics):
        self.exp_logger.log_metrics(metrics, step=self.epoch)

    def setup_bookkeeping(self):
        pass

    @property
    def event_handlers_train(self):
        return [
            (Events.STARTED, self.load_state),
            (Events.EPOCH_COMPLETED, self.save_state),
            (Events.COMPLETED, self.save_results)]

    def load_state(self):
        objs_and_state_files = [
            (self.train_engine.state, self.conf.trainer_state_file),
            (self.checkpointer, self.conf.checkpointer_state_file)]
        for obj, state_file in objs_and_state_files:
            if state_file.exists():
                self.log(f'loading {state_file}')
                if state_file.suffix == '.pt':
                    obj.load_state_dict(torch.load(state_file))
                else:
                    state = json_load(state_file)
                    for k, v in state.items():
                        setattr(obj, k, v)
                        self.log(f'{k}: {getattr(obj, k)}')
        checkpoint_file = self.last_checkpoint_file
        if checkpoint_file:
            checkpoint = torch.load(checkpoint_file)
            for k, v in checkpoint.items():
                if k == 'model':
                    v = fix_dataparallel_statedict(self.model, v)
                getattr(self, k).load_state_dict(v)
                self.log(f'loaded state dict: {k}')

    def save_state(self, engine=None):
        trainer_state = dict(
            epoch=engine.state.epoch,
            iteration=engine.state.iteration,
            metrics=engine.state.metrics
            )
        json_dump(trainer_state, self.conf.trainer_state_file)
        torch.save(
            self.checkpointer.state_dict(), self.conf.checkpointer_state_file)

    def save_results(self, engine=None):
        checkpoint_file = self.best_checkpoint or 'no_checkpoint'
        params = self.exp_logger.exp_params
        run_info = dict(
            checkpoint_file=str(checkpoint_file),
            final_epoch=self.train_engine.state.epoch)
        self.exp_logger.log_params(run_info)
        self.exp_logger.log_artifacts()
        metrics = self.eval_engine.state.metrics
        self.results = dict(**params, **run_info, **metrics)
        print(self.results)
        fname = (self.conf.runid or 'results') + '.json'
        results_file = mkdir(self.conf.outdir / 'results') / fname
        json_dump(self.results, results_file)
        self.log(results_file)

    def save_stdout(self):
        jobid = getattr(self.conf, 'jobid', None)
        if jobid and hasattr(self.conf, 'stdout_dir'):
            stdout_file = (self.conf.stdout_dir / str(jobid)).expanduser()
            stdout_copy = self.conf.rundir / 'stdout.txt'
            try:
                import shutil
                shutil.copy(str(stdout_file), str(stdout_copy))
            except Exception:
                pass

    def cleanup(self):
        if self.conf.distributed and self.is_dist_main():
            # Pytorch distributed processes continue running after the main
            # process has finished, which means the cluster node we're running
            # on doesn't get freed up.
            # try to kill those manually, might kill other python processes
            # if we're not running exclusively ¯\_(ツ)_/¯
            dist.destroy_process_group()
            main_pid = os.getpid()
            output = subprocess.check_output("pidof -c python".split())
            pids = list(map(int, output.split()))
            for pid in pids:
                if pid > main_pid:
                    os.kill(pid, 9)

    def train(self):
        self.start_run()
        self.train_engine.run(
            self.data.train_loader, max_epochs=self.conf.max_epochs)
        self.end_run()
        if self.is_dist_main():
            return self.results

    def test(self):
        self.start_run()
        self.eval_engine.run(self.data.test_loader)
        self.log_metrics(self.eval_engine.state.metrics)
        self.log_results('test', self.eval_engine.state.metrics)
        self.save_results()
        self.end_run()
        if self.is_dist_main():
            return self.results

    def dev(self):
        breakpoint()

    def cache_dataset(self):
        self.load_data()

    def start_run(self):
        if self.is_dist_main():
            self.exp_logger.start_run()

    def end_run(self):
        if self.is_dist_main():
            self.save_stdout()
            self.exp_logger.end_run()
        self.cleanup()
