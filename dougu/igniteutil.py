from collections import defaultdict
from typing import Callable

from dougu.torchutil import get_lr_scheduler

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping


def attach_lr_scheduler(
        engine, optim, conf,
        event=Events.COMPLETED,
        metric_name='acc',
        optimum='max',
        n_train_steps=None,
        log=None,
        lr_scheduler=None):
    if lr_scheduler is None:
        lr_scheduler = get_lr_scheduler(
            conf, optim, optimum=optimum, n_train_steps=n_train_steps)

    if lr_scheduler is not None:
        @engine.on(event)
        def scheduler_step(evaluator):
            try:
                if lr_scheduler.requires_metric:
                    lr_scheduler.step(evaluator.state.metrics[metric_name])
                else:
                    lr_scheduler.step()
            except Exception:
                import traceback
                traceback.print_exc()
    return lr_scheduler


def log_results(trainer, evaluator, eval_name):
    metrics = evaluator.state.metrics
    if not hasattr(trainer.state, 'last_scores'):
        trainer.state.last_scores = {}
    metrics_str = ' | '.join(
        f'{metric} {val:.4f}' for metric, val in metrics.items())
    trainer.log("epoch {:04d} {} {} | {}".format(
            trainer.state.epoch,
            eval_name,
            trainer.state.last_epoch_duration,
            metrics_str))
    for metric_name, metric_score in metrics.items():
        if metric_name == 'nll':
            continue
        attr_name = f'{eval_name}_{metric_name}'
        trainer.state.last_scores[attr_name] = metric_score
    if 'acc' in metrics:
        trainer.state.last_acc = metrics['acc']


def attach_result_log(
        trainer,
        evaluators,
        data_loaders,
        eval_every=1,
        first_eval_epoch=1,
        ):
    def _log_results(_trainer):
        if _trainer.state.epoch < first_eval_epoch:
            return
        for split_name, evaluator in evaluators.items():
            data_loader = data_loaders[split_name]
            evaluator.run(data_loader)
            log_results(_trainer, evaluator, split_name)

    eval_event = Events.EPOCH_COMPLETED(every=eval_every)
    trainer.add_event_handler(eval_event, _log_results)


def make_trainer(
        name='trainer',
        optim=None,
        conf=None,
        log=None,
        metrics=None,
        metric_name='loss'):
    """Decorator that turns an ignite update function into a training
    engine creation function.
    """
    def actual_decorator(update_func):
        def wrapper(*args, **kwargs):
            engine = Engine(update_func, name=name)
            if metrics:
                for metric_name, metric in metrics.items():
                    metric.attach(engine, metric_name)
            if conf and optim and conf.learning_rate_scheduler != 'plateau':
                attach_lr_scheduler(
                    engine, optim, conf, log=log,
                    metric_name=metric_name,
                    event=Events.ITERATION_COMPLETED)
            return engine
        return wrapper
    return actual_decorator


def make_evaluator(
        metrics, optim, conf, lr_metric='acc', optimum='max'):
    """Decorator that turns an ignite inference function into a test
    engine creation function.
    """
    def actual_decorator(inference_func):
        def wrapper(*args, **kwargs):
            engine = Engine(inference_func)
            if conf.learning_rate_scheduler == 'plateau':
                attach_lr_scheduler(
                    engine, optim, conf,
                    metric_name=lr_metric, optimum=optimum)

            @engine.on(Events.STARTED)
            def reset_io(engine):
                engine.state.io = defaultdict(list)

            for name, metric in metrics.items():
                metric.attach(engine, name)
            return engine
        return wrapper
    return actual_decorator


def attach_checkpointer(
        to_save, evaluator, *, rundir,
        checkpoint_metric='acc',
        checkpoint_metric_optimum='max',
        checkpoint_prefix='',
        first_save_after=0):
    # pbar = ProgressBar()
    # pbar.attach(trainer)
    if checkpoint_metric:
        sign = {
            'max': 1,
            'min': -1}[checkpoint_metric_optimum]
        checkpointer = ModelCheckpoint(
            rundir,
            checkpoint_prefix,
            score_name=checkpoint_metric,
            score_function=lambda _: sign * evaluator.state.metrics[
                checkpoint_metric],
            n_saved=3,
            require_empty=False,
            first_save_after=first_save_after)
        if not isinstance(to_save, dict):
            to_save = {'model': to_save}
        evaluator.add_event_handler(
            Events.COMPLETED, checkpointer, to_save)
        return checkpointer


class EarlyStoppingWithBurnin(EarlyStopping):
    def __init__(
        self,
        patience: int,
        score_function: Callable,
        trainer: Engine,
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
        burnin: int = 0,
    ):
        super().__init__(
            patience,
            score_function,
            trainer,
            min_delta=min_delta,
            cumulative_delta=cumulative_delta,
        )
        self.burnin = burnin
        self._steps = 0

    def __call__(self, engine: Engine) -> None:
        self._steps += 1
        if self._steps > self.burnin:
            super().__call__(engine)
