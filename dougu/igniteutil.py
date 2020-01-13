from collections import defaultdict

from dougu.torchutil import get_lr_scheduler

from .ignite import Engine, Events
from .ignite.handlers import ModelCheckpoint
from .ignite.contrib.handlers import CustomPeriodicEvent, ProgressBar


def attach_lr_scheduler(
        engine, optim, conf,
        event=Events.COMPLETED,
        metric_name='acc', optimum='max', t_total=None, log=None):
    lr_scheduler = get_lr_scheduler(
        conf, optim, optimum=optimum, t_total=t_total)

    if lr_scheduler is not None and metric_name is not None:
        @engine.on(event)
        def scheduler_step(evaluator):
            if log:
                log(f'lr: {lr_scheduler.get_lr()}')
            try:
                if metric_name:
                    lr_scheduler.step(evaluator.state.metrics[metric_name])
                else:
                    lr_scheduler.step()
            except Exception:
                import traceback
                traceback.print_exc()


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
        trainer, evaluators, data_loaders,
        eval_every=1):
    def _log_results(_trainer):
        for split_name, evaluator in evaluators.items():
            data_loader = data_loaders[split_name]
            evaluator.run(data_loader)
            log_results(_trainer, evaluator, split_name)

    if eval_every != 1:
        eval_events = CustomPeriodicEvent(n_epochs=eval_every)
        trainer.register_events(*eval_events.Events)
        eval_events.attach(trainer)
        eval_event = getattr(
            eval_events.Events, f'EPOCHS_{eval_every}_COMPLETED')
    else:
        eval_event = Events.EPOCH_COMPLETED
    trainer.add_event_handler(eval_event, _log_results)


def make_trainer(
        name='trainer',
        optim=None,
        conf=None,
        log=None,
        metric_name='loss'):
    """Decorator that turns an ignite update function into a training
    engine creation function.
    """
    def actual_decorator(update_func):
        def wrapper(*args, **kwargs):
            engine = Engine(update_func, name=name)
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
    creation function.
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
        model, evaluator, *, rundir,
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
        evaluator.add_event_handler(
            Events.COMPLETED, checkpointer, {'model': model})
