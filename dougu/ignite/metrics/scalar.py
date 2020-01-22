from __future__ import division

from ..exceptions import NotComputableError
from .metric import Metric


class Scalar(Metric):
    """
    Calculates the average loss according to the passed loss_fn.

    Args:
        loss_fn (callable): a callable taking a prediction tensor, a target
            tensor, optionally other arguments, and returns the average loss
            over all observations in the batch.
        output_transform (callable): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric.
            This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            The output is is expected to be a tuple (prediction, target) or
            (prediction, target, kwargs) where kwargs is a dictionary of extra
            keywords arguments.
        batch_size (callable): a callable taking a target tensor that returns the
            first dimension size (usually the batch size).

    """

    def __init__(self, loss_fn, output_transform=lambda x: x,
                 batch_size=lambda x: len(x)):
        super(Scalar, self).__init__(output_transform)
        self._loss_fn = loss_fn
        self._batch_size = batch_size

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        if len(output) == 2:
            dummy, scalar = output
            kwargs = {}
        else:
            dummy, scalar, kwargs = output

        self._sum += scalar
        self._num_examples += 1

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')
        return self._sum / self._num_examples
