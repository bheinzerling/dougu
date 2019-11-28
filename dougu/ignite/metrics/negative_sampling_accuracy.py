from __future__ import division

from ..exceptions import NotComputableError
from .metric import Metric

import torch


class NegativeSamplingAccuracy(Metric):
    """
    When training with positive and negative samples, this metric
    calculates the percentage of scores of positive samples which
    are higher than scores of negative samples.

    Args:
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

    def __init__(self, output_transform=lambda x: x,
                 batch_size=lambda x: len(x)):
        super(NegativeSamplingAccuracy, self).__init__(output_transform)
        self._batch_size = batch_size

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        pos_scores, neg_scores = output
        is_higher = (pos_scores.unsqueeze(dim=1) > neg_scores)
        n_higher = is_higher.to(dtype=torch.float).sum(dim=1)
        self._sum += n_higher.mean()
        print(n_higher.mean())
        self._num_examples += 1

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')
        return self._sum / self._num_examples
