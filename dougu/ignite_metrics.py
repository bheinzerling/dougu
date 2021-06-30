import numbers
import itertools
from typing import Any, Callable, Optional, Sequence, Union

import torch

from ignite.engine import Engine

from ignite.metrics import Metric, MetricUsage, EpochWise
from ignite.utils import to_onehot
from ignite.exceptions import NotComputableError


class _BaseClassification(Metric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        is_multilabel: bool = False,
        device: Optional[Union[str, torch.device]] = 'cpu',
        ignore_idx: int = -100,
    ):
        self._is_multilabel = is_multilabel
        self._type = None
        self._num_classes = None
        self.ignore_idx = ignore_idx
        super().__init__(output_transform=output_transform, device=device)

    def reset(self) -> None:
        self._type = None
        self._num_classes = None

    def filter_ignored(self, y_pred, y):
        if self.ignore_idx is not None:
            ignore_mask = y != self.ignore_idx
            y = y[ignore_mask]
            y_pred = y_pred[ignore_mask]
        return y_pred, y

    def _check_shape(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError(
                "y must have shape of (batch_size, ...) and y_pred must have "
                "shape of (batch_size, num_categories, ...) or (batch_size, ...), "
                "but given {} vs {}.".format(y.shape, y_pred.shape)
            )

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        if self._is_multilabel and not (y.shape == y_pred.shape and y.ndimension() > 1 and y.shape[1] > 1):
            raise ValueError(
                "y and y_pred must have same shape of (batch_size, num_categories, ...) and num_categories > 1."
            )

    def _check_binary_multilabel_cases(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output

        if not torch.equal(y, y ** 2):
            raise ValueError("For binary cases, y must be comprised of 0's and 1's.")

        if not torch.equal(y_pred, y_pred ** 2):
            raise ValueError("For binary cases, y_pred must be comprised of 0's and 1's.")

    def _check_type(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output

        if y.ndimension() + 1 == y_pred.ndimension():
            num_classes = y_pred.shape[1]
            if num_classes == 1:
                update_type = "binary"
                self._check_binary_multilabel_cases((y_pred, y))
            else:
                update_type = "multiclass"
        elif y.ndimension() == y_pred.ndimension():
            self._check_binary_multilabel_cases((y_pred, y))

            if self._is_multilabel:
                update_type = "multilabel"
                num_classes = y_pred.shape[1]
            else:
                update_type = "binary"
                num_classes = 1
        else:
            raise RuntimeError(
                "Invalid shapes of y (shape={}) and y_pred (shape={}), check documentation."
                " for expected shapes of y and y_pred.".format(y.shape, y_pred.shape)
            )
        if self._type is None:
            self._type = update_type
            self._num_classes = num_classes
        else:
            if self._type != update_type:
                raise RuntimeError("Input data type has changed from {} to {}.".format(self._type, update_type))
            if self._num_classes != num_classes:
                raise ValueError(
                    "Input data number of classes has changed from {} to {}".format(self._num_classes, num_classes)
                )


class MetricsLambda(Metric):
    """
    Apply a function to other metrics to obtain a new metric.
    The result of the new metric is defined to be the result
    of applying the function to the result of argument metrics.
    When update, this metric does not recursively update the metrics
    it depends on. When reset, all its dependency metrics would be
    resetted. When attach, all its dependency metrics would be attached
    automatically (but partially, e.g :meth:`~ignite.metrics.Metric.is_attached()` will return False).
    Args:
        f (callable): the function that defines the computation
        args (sequence): Sequence of other metrics or something
            else that will be fed to ``f`` as arguments.
    Example:
    .. code-block:: python
        precision = Precision(average=False)
        recall = Recall(average=False)
        def Fbeta(r, p, beta):
            return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()
        F1 = MetricsLambda(Fbeta, recall, precision, 1)
        F2 = MetricsLambda(Fbeta, recall, precision, 2)
        F3 = MetricsLambda(Fbeta, recall, precision, 3)
        F4 = MetricsLambda(Fbeta, recall, precision, 4)
    When check if the metric is attached, if one of its dependency
    metrics is detached, the metric is considered detached too.
    .. code-block:: python
        engine = ...
        precision = Precision(average=False)
        aP = precision.mean()
        aP.attach(engine, "aP")
        assert aP.is_attached(engine)
        # partially attached
        assert not precision.is_attached(engine)
        precision.detach(engine)
        assert not aP.is_attached(engine)
        # fully attached
        assert not precision.is_attached(engine)
    """

    def __init__(self, f: Callable, *args, **kwargs):
        self.function = f
        self.args = args
        self.kwargs = kwargs
        self.engine = None
        super(MetricsLambda, self).__init__(device="cpu")

    def reset(self) -> None:
        for i in itertools.chain(self.args, self.kwargs.values()):
            if isinstance(i, Metric):
                i.reset()

    def update(self, output) -> None:
        # NB: this method does not recursively update dependency metrics,
        # which might cause duplicate update issue. To update this metric,
        # users should manually update its dependencies.
        pass

    def compute(self) -> Any:
        materialized = [i.compute() if isinstance(i, Metric) else i for i in self.args]
        materialized_kwargs = {k: (v.compute() if isinstance(v, Metric) else v) for k, v in self.kwargs.items()}
        return self.function(*materialized, **materialized_kwargs)

    def _internal_attach(self, engine: Engine, usage: MetricUsage) -> None:
        self.engine = engine
        for index, metric in enumerate(itertools.chain(self.args, self.kwargs.values())):
            if isinstance(metric, MetricsLambda):
                metric._internal_attach(engine, usage)
            elif isinstance(metric, Metric):
                # NB : metrics is attached partially
                # We must not use is_attached() but rather if these events exist
                if not engine.has_event_handler(metric.started, usage.STARTED):
                    engine.add_event_handler(usage.STARTED, metric.started)
                if not engine.has_event_handler(metric.iteration_completed, usage.ITERATION_COMPLETED):
                    engine.add_event_handler(usage.ITERATION_COMPLETED, metric.iteration_completed)

    def attach(self, engine: Engine, name: str, usage: Union[str, MetricUsage] = EpochWise()) -> None:
        usage = self._check_usage(usage)
        # recursively attach all its dependencies (partially)
        self._internal_attach(engine, usage)
        # attach only handler on EPOCH_COMPLETED
        engine.add_event_handler(usage.COMPLETED, self.completed, name)

    def detach(self, engine: Engine, usage: Union[str, MetricUsage] = EpochWise()) -> None:
        usage = self._check_usage(usage)
        # remove from engine
        super(MetricsLambda, self).detach(engine, usage)
        self.engine = None

    def is_attached(self, engine: Engine, usage: Union[str, MetricUsage] = EpochWise()) -> bool:
        usage = self._check_usage(usage)
        # check recursively the dependencies
        return super(MetricsLambda, self).is_attached(engine, usage) and self._internal_is_attached(engine, usage)

    def _internal_is_attached(self, engine: Engine, usage: MetricUsage) -> bool:
        # if no engine, metrics is not attached
        if engine is None:
            return False
        # check recursively if metrics are attached
        is_detached = False
        for metric in itertools.chain(self.args, self.kwargs.values()):
            if isinstance(metric, MetricsLambda):
                if not metric._internal_is_attached(engine, usage):
                    is_detached = True
            elif isinstance(metric, Metric):
                if not engine.has_event_handler(metric.started, usage.STARTED):
                    is_detached = True
                if not engine.has_event_handler(metric.iteration_completed, usage.ITERATION_COMPLETED):
                    is_detached = True
        return not is_detached


class _BasePrecisionRecall(_BaseClassification):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        average: bool = False,
        is_multilabel: bool = False,
        device: Optional[Union[str, torch.device]] = 'cpu',
        ignore_idx: int = -100,
    ):

        self._average = average
        self._true_positives = None
        self._positives = None
        self.eps = 1e-20
        super().__init__(
            output_transform=output_transform, is_multilabel=is_multilabel, device=device
        )

    def reset(self) -> None:
        dtype = torch.float32
        self._true_positives = torch.tensor([], dtype=dtype) if (self._is_multilabel and not self._average) else 0
        self._positives = torch.tensor([], dtype=dtype) if (self._is_multilabel and not self._average) else 0
        super().reset()

    def compute(self) -> Union[torch.Tensor, float]:
        if not (isinstance(self._positives, torch.Tensor) or self._positives > 0):
            raise NotComputableError(
                "{} must have at least one example before" " it can be computed.".format(self.__class__.__name__)
            )

        result = self._true_positives / (self._positives + self.eps)

        if self._average:
            return result.mean().item()
        else:
            return result


class VariableAccumulation(Metric):
    """Single variable accumulator helper to compute (arithmetic, geometric, harmonic) average of a single variable.
    - ``update`` must receive output of the form `x`.
    - `x` can be a number or `torch.Tensor`.
    Note:
        The class stores input into two public variables: `accumulator` and `num_examples`.
        Number of samples is updated following the rule:
        - `+1` if input is a number
        - `+1` if input is a 1D `torch.Tensor`
        - `+batch_size` if input is a ND `torch.Tensor`. Batch size is the first dimension (`shape[0]`).
    Args:
        op (callable): a callable to update accumulator. Method's signature is `(accumulator, output)`.
            For example, to compute arithmetic mean value, `op = lambda a, x: a + x`.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device (str of torch.device, optional): optional device specification for internal storage.
    """

    _required_output_keys = None

    def __init__(
        self, op: Callable, output_transform: Callable = lambda x: x, device: Optional[Union[str, torch.device]] = 'cpu'
    ):
        if not callable(op):
            raise TypeError("Argument op should be a callable, but given {}".format(type(op)))
        self.accumulator = None
        self.num_examples = None
        self._op = op

        super(VariableAccumulation, self).__init__(output_transform=output_transform, device=device)

    def reset(self) -> None:
        self.accumulator = torch.tensor(0.0, dtype=torch.float32, device=self._device)
        self.num_examples = torch.tensor(0, dtype=torch.long, device=self._device)

    def _check_output_type(self, output: Union[Any, torch.Tensor, numbers.Number]) -> None:
        if not (isinstance(output, numbers.Number) or isinstance(output, torch.Tensor)):
            raise TypeError("Output should be a number or torch.Tensor, but given {}".format(type(output)))

    def update(self, output: Union[Any, torch.Tensor, numbers.Number]) -> None:
        self._check_output_type(output)

        if self._device is not None:
            # Put output to the metric's device
            if isinstance(output, torch.Tensor) and (output.device != self._device):
                output = output.to(self._device)

        self.accumulator = self._op(self.accumulator, output)
        if hasattr(output, "shape"):
            self.num_examples += output.shape[0] if len(output.shape) > 1 else 1
        else:
            self.num_examples += 1

    def compute(self) -> list:
        return [self.accumulator, self.num_examples]


# Same as original Ignite implementation, except no @sync_all_reduce.
# We remove this since we're calculating metrics in the main process only,
# which leads to @sync_all_reduce to hang because the processes it is
# waiting for do not exist
class Average(VariableAccumulation):
    """Helper class to compute arithmetic average of a single variable.
    - ``update`` must receive output of the form `x`.
    - `x` can be a number or `torch.Tensor`.
    Note:
        Number of samples is updated following the rule:
        - `+1` if input is a number
        - `+1` if input is a 1D `torch.Tensor`
        - `+batch_size` if input is an ND `torch.Tensor`. Batch size is the first dimension (`shape[0]`).
        For input `x` being an ND `torch.Tensor` with N > 1, the first dimension is seen as the number of samples and
        is summed up and added to the accumulator: `accumulator += x.sum(dim=0)`
    Examples:
    .. code-block:: python
        evaluator = ...
        custom_var_mean = Average(output_transform=lambda output: output['custom_var'])
        custom_var_mean.attach(evaluator, 'mean_custom_var')
        state = evaluator.run(dataset)
        # state.metrics['mean_custom_var'] -> average of output['custom_var']
    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device (str of torch.device, optional): optional device specification for internal storage.
    """

    def __init__(self, output_transform: Callable = lambda x: x, device: Optional[Union[str, torch.device]] = 'cpu'):
        def _mean_op(a, x):
            if isinstance(x, torch.Tensor) and x.ndim > 1:
                x = x.sum(dim=0)
            return a + x

        super(Average, self).__init__(op=_mean_op, output_transform=output_transform, device=device)

    def compute(self) -> Union[Any, torch.Tensor, numbers.Number]:
        if self.num_examples < 1:
            raise NotComputableError(
                "{} must have at least one example before" " it can be computed.".format(self.__class__.__name__)
            )

        return self.accumulator / self.num_examples


class Accuracy(_BaseClassification):
    """
    Calculates the accuracy for binary, multiclass and multilabel data.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).
    - `y` and `y_pred` must be in the following shape of (batch_size, num_categories, ...) and
      num_categories must be greater than 1 for multilabel cases.

    In binary and multilabel cases, the elements of `y` and `y_pred` should have 0 or 1 values. Thresholding of
    predictions can be done as below:

    .. code-block:: python

        def thresholded_output_transform(output):
            y_pred, y = output
            y_pred = torch.round(y_pred)
            return y_pred, y

        binary_accuracy = Accuracy(thresholded_output_transform)


    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        is_multilabel (bool, optional): flag to use in multilabel case. By default, False.
        device (str of torch.device, optional): unused argument.

    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        is_multilabel: bool = False,
        device: Optional[Union[str, torch.device]] = 'cpu',
        ignore_idx: int = -100,
    ):
        self._num_correct = None
        self._num_examples = None
        super(Accuracy, self).__init__(output_transform=output_transform, is_multilabel=is_multilabel, device=device, ignore_idx=ignore_idx)

    def reset(self) -> None:
        self._num_correct = 0
        self._num_examples = 0
        super(Accuracy, self).reset()

    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        y_pred, y = self.filter_ignored(y_pred, y)
        self._check_shape((y_pred, y))
        self._check_type((y_pred, y))

        if self._type == "binary":
            correct = torch.eq(y_pred.view(-1).to(y), y.view(-1))
        elif self._type == "multiclass":
            indices = torch.argmax(y_pred, dim=1)
            correct = torch.eq(indices, y).view(-1)
        elif self._type == "multilabel":
            # if y, y_pred shape is (N, C, ...) -> (N x ..., C)
            num_classes = y_pred.size(1)
            last_dim = y_pred.ndimension()
            y_pred = torch.transpose(y_pred, 1, last_dim - 1).reshape(-1, num_classes)
            y = torch.transpose(y, 1, last_dim - 1).reshape(-1, num_classes)
            correct = torch.all(y == y_pred.type_as(y), dim=-1)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("Accuracy must have at least one example before it can be computed.")
        return self._num_correct / self._num_examples


class TopKCategoricalAccuracy(Metric):
    """
    Calculates the top-k categorical accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def __init__(
            self, k=5, output_transform=lambda x: x, already_sorted=False):
        super(TopKCategoricalAccuracy, self).__init__(output_transform)
        self._k = k
        self.already_sorted = already_sorted

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        if self.already_sorted:
            sorted_indices = y_pred[..., :self._k]
        else:
            sorted_indices = torch.topk(y_pred, self._k, dim=-1)[1]
        sorted_indices = sorted_indices.view(-1, self._k)
        expanded_y = y.view(-1, 1).expand(-1, self._k)
        correct = torch.sum(torch.eq(sorted_indices, expanded_y), dim=1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("TopKCategoricalAccuracy must have at"
                                     "least one example before it can be computed.")
        return self._num_correct / self._num_examples


class Recall(_BasePrecisionRecall):
    """
    Calculates recall for binary and multiclass data.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).

    In binary and multilabel cases, the elements of `y` and `y_pred` should have 0 or 1 values. Thresholding of
    predictions can be done as below:

    .. code-block:: python

        def thresholded_output_transform(output):
            y_pred, y = output
            y_pred = torch.round(y_pred)
            return y_pred, y

        recall = Recall(output_transform=thresholded_output_transform)

    In multilabel cases, average parameter should be True. However, if user would like to compute F1 metric, for
    example, average parameter should be False. This can be done as shown below:

    .. code-block:: python

        precision = Precision(average=False, is_multilabel=True)
        recall = Recall(average=False, is_multilabel=True)
        F1 = precision * recall * 2 / (precision + recall + 1e-20)
        F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)

    .. warning::

        In multilabel cases, if average is False, current implementation stores all input data (output and target) in
        as tensors before computing a metric. This can potentially lead to a memory error if the input data is larger
        than available RAM.

    .. warning::

        In multilabel cases, if average is False, current implementation does not work with distributed computations.
        Results are not reduced across the GPUs. Computed result corresponds to the local rank's (single GPU) result.


    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        average (bool, optional): if True, precision is computed as the unweighted average (across all classes
            in multiclass case), otherwise, returns a tensor with the precision (for each class in multiclass case).
        is_multilabel (bool, optional) flag to use in multilabel case. By default, value is False. If True, average
            parameter should be True and the average is computed across samples, instead of classes.
        device (str of torch.device, optional): unused argument.

    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        average: bool = False,
        is_multilabel: bool = False,
        device: Optional[Union[str, torch.device]] = 'cpu',
        ignore_idx: int = -100,
    ):
        super(Recall, self).__init__(
            output_transform=output_transform, average=average, is_multilabel=is_multilabel, device=device, ignore_idx=ignore_idx,
        )

    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        self._check_shape(output)
        y_pred, y = self.filter_ignored(y_pred, y)
        self._check_type((y_pred, y))

        if self._type == "binary":
            y_pred = y_pred.view(-1)
            y = y.view(-1)
        elif self._type == "multiclass":
            num_classes = y_pred.size(1)
            if y.max() + 1 > num_classes:
                raise ValueError(
                    "y_pred contains less classes than y. Number of predicted classes is {}"
                    " and element in y has invalid class = {}.".format(num_classes, y.max().item() + 1)
                )
            y = to_onehot(y.view(-1), num_classes=num_classes)
            indices = torch.argmax(y_pred, dim=1).view(-1)
            y_pred = to_onehot(indices, num_classes=num_classes)
        elif self._type == "multilabel":
            # if y, y_pred shape is (N, C, ...) -> (C, N x ...)
            num_classes = y_pred.size(1)
            y_pred = torch.transpose(y_pred, 1, 0).reshape(num_classes, -1)
            y = torch.transpose(y, 1, 0).reshape(num_classes, -1)

        y = y.type_as(y_pred)
        correct = y * y_pred
        # actual_positives = y.sum(dim=0).to(dtype=torch.float32)  # Convert from int cuda/cpu to double cpu
        actual_positives = y.sum(dim=0)

        if correct.sum() == 0:
            true_positives = torch.zeros_like(actual_positives)
        else:
            true_positives = correct.sum(dim=0)

        if self._type == "multilabel":
            if not self._average:
                self._true_positives = torch.cat([self._true_positives, true_positives], dim=0)
                self._positives = torch.cat([self._positives, actual_positives], dim=0)
            else:
                self._true_positives += torch.sum(true_positives / (actual_positives + self.eps))
                self._positives += len(actual_positives)
        else:
            self._true_positives += true_positives
            self._positives += actual_positives


class _BasePrecisionRecall(_BaseClassification):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        average: bool = False,
        is_multilabel: bool = False,
        device: Optional[Union[str, torch.device]] = 'cpu',
        ignore_idx: int = -100,
    ):
        self._average = average
        self._true_positives = None
        self._positives = None
        self.eps = 1e-20
        super().__init__(
            output_transform=output_transform, is_multilabel=is_multilabel, device=device, ignore_idx=ignore_idx,
        )

    def reset(self) -> None:
        dtype = torch.float32
        self._true_positives = torch.tensor([], dtype=dtype) if (self._is_multilabel and not self._average) else 0
        self._positives = torch.tensor([], dtype=dtype) if (self._is_multilabel and not self._average) else 0
        super(_BasePrecisionRecall, self).reset()

    def compute(self) -> torch.Tensor:
        if not (isinstance(self._positives, torch.Tensor) or self._positives > 0):
            raise NotComputableError(
                "{} must have at least one example before" " it can be computed.".format(self.__class__.__name__)
            )

        result = self._true_positives / (self._positives + self.eps)

        if self._average:
            return result.mean().item()
        else:
            return result


class Precision(_BasePrecisionRecall):
    """
    Calculates precision for binary and multiclass data.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).

    In binary and multilabel cases, the elements of `y` and `y_pred` should have 0 or 1 values. Thresholding of
    predictions can be done as below:

    .. code-block:: python

        def thresholded_output_transform(output):
            y_pred, y = output
            y_pred = torch.round(y_pred)
            return y_pred, y

        precision = Precision(output_transform=thresholded_output_transform)

    In multilabel cases, average parameter should be True. However, if user would like to compute F1 metric, for
    example, average parameter should be False. This can be done as shown below:

    .. code-block:: python

        precision = Precision(average=False, is_multilabel=True)
        recall = Recall(average=False, is_multilabel=True)
        F1 = precision * recall * 2 / (precision + recall + 1e-20)
        F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)

    .. warning::

        In multilabel cases, if average is False, current implementation stores all input data (output and target) in
        as tensors before computing a metric. This can potentially lead to a memory error if the input data is larger
        than available RAM.

    .. warning::

        In multilabel cases, if average is False, current implementation does not work with distributed computations.
        Results are not reduced across the GPUs. Computed result corresponds to the local rank's (single GPU) result.


    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        average (bool, optional): if True, precision is computed as the unweighted average (across all classes
            in multiclass case), otherwise, returns a tensor with the precision (for each class in multiclass case).
        is_multilabel (bool, optional) flag to use in multilabel case. By default, value is False. If True, average
            parameter should be True and the average is computed across samples, instead of classes.
        device (str of torch.device, optional): unused argument.

    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        average: bool = False,
        is_multilabel: bool = False,
        device: Optional[Union[str, torch.device]] = 'cpu',
        ignore_idx: int = -100,
    ):
        super(Precision, self).__init__(
            output_transform=output_transform, average=average, is_multilabel=is_multilabel, device=device, ignore_idx=ignore_idx,
        )

    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        self._check_shape(output)
        y_pred, y = self.filter_ignored(y_pred, y)
        self._check_type((y_pred, y))

        if self._type == "binary":
            y_pred = y_pred.view(-1)
            y = y.view(-1)
        elif self._type == "multiclass":
            num_classes = y_pred.size(1)
            if y.max() + 1 > num_classes:
                raise ValueError(
                    "y_pred contains less classes than y. Number of predicted classes is {}"
                    " and element in y has invalid class = {}.".format(num_classes, y.max().item() + 1)
                )
            y = to_onehot(y.view(-1), num_classes=num_classes)
            indices = torch.argmax(y_pred, dim=1).view(-1)
            y_pred = to_onehot(indices, num_classes=num_classes)
        elif self._type == "multilabel":
            # if y, y_pred shape is (N, C, ...) -> (C, N x ...)
            num_classes = y_pred.size(1)
            y_pred = torch.transpose(y_pred, 1, 0).reshape(num_classes, -1)
            y = torch.transpose(y, 1, 0).reshape(num_classes, -1)

        y = y.to(y_pred)
        correct = y * y_pred
        all_positives = y_pred.sum(dim=0).to(dtype=torch.float32)  # Convert from int cuda/cpu to double cpu

        if correct.sum() == 0:
            true_positives = torch.zeros_like(all_positives)
        else:
            true_positives = correct.sum(dim=0)
        # Convert from int cuda/cpu to double cpu
        # We need double precision for the division true_positives / all_positives
        true_positives = true_positives.to(dtype=torch.float32)

        if self._type == "multilabel":
            if not self._average:
                self._true_positives = torch.cat([self._true_positives, true_positives], dim=0)
                self._positives = torch.cat([self._positives, all_positives], dim=0)
            else:
                self._true_positives += torch.sum(true_positives / (all_positives + self.eps))
                self._positives += len(all_positives)
        else:
            self._true_positives += true_positives
            self._positives += all_positives


def Fbeta(
    beta: float,
    average: bool = True,
    precision: Optional[Precision] = None,
    recall: Optional[Recall] = None,
    output_transform: Optional[Callable] = None,
    device: Optional[Union[str, torch.device]] = 'cpu',
) -> MetricsLambda:
    """Calculates F-beta score

    Args:
        beta (float): weight of precision in harmonic mean
        average (bool, optional): if True, F-beta score is computed as the unweighted average (across all classes
            in multiclass case), otherwise, returns a tensor with F-beta score for each class in multiclass case.
        precision (Precision, optional): precision object metric with `average=False` to compute F-beta score
        recall (Precision, optional): recall object metric with `average=False` to compute F-beta score
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. It is used only if precision or recall are not provided.
        device (str of torch.device, optional): optional device specification for internal storage.

    Returns:
        MetricsLambda, F-beta metric
    """
    if not (beta > 0):
        raise ValueError("Beta should be a positive integer, but given {}".format(beta))

    if precision is not None and output_transform is not None:
        raise ValueError("If precision argument is provided, output_transform should be None")

    if recall is not None and output_transform is not None:
        raise ValueError("If recall argument is provided, output_transform should be None")

    if precision is None:
        precision = Precision(
            output_transform=(lambda x: x) if output_transform is None else output_transform,
            average=False,
            device=device,
        )
    elif precision._average:
        raise ValueError("Input precision metric should have average=False")

    if recall is None:
        recall = Recall(
            output_transform=(lambda x: x) if output_transform is None else output_transform,
            average=False,
            device=device,
        )
    elif recall._average:
        raise ValueError("Input recall metric should have average=False")

    fbeta = (1.0 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-15)

    if average:
        fbeta = fbeta.mean().item()

    return fbeta


class MeanAbsoluteError(Metric):
    """
    Calculates the mean absolute error.
    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    """

    def reset(self) -> None:
        self._sum_of_absolute_errors = 0.0
        self._num_examples = 0

    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        absolute_errors = torch.abs(y_pred - y.view_as(y_pred))
        self._sum_of_absolute_errors += torch.sum(absolute_errors).item()
        self._num_examples += y.shape[0]

    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError("MeanAbsoluteError must have at least one example before it can be computed.")
        return self._sum_of_absolute_errors / self._num_examples


class MeanReciprocalRank(Metric):

    def __init__(self, *args, mode='prob', **kwargs):
        super().__init__(*args, **kwargs)
        self.update = {
            'prob': self.update_prob,
            'idx': self.update_idx}[mode]

    def reset(self, mode='prob'):
        self.ranks = []

    def update(self, output):
        raise NotImplementedError()

    def update_prob(self, output):
        probs, target = output
        if (probs < 0).any():
            assert (probs <= 0).all()
            probs = torch.exp(probs)
        correct_prob = probs.gather(1, target.unsqueeze(1))
        rank = (probs >= correct_prob).sum(dim=1)
        self.ranks.append(rank.cpu())

    def update_idx(self, output):
        pred_idx, target = output
        matches = pred_idx == target.unsqueeze(1)
        bs, max_rank = pred_idx.shape
        rank = torch.zeros(bs).long() + max_rank
        match_batch_idxs, match_rank = matches.nonzero().t().cpu()
        # + 1 to convert from zero-indexed to rank
        rank[match_batch_idxs] = match_rank + 1
        self.ranks.append(rank)

    def compute(self):
        rank = torch.cat(self.ranks)
        return (1 / rank.float()).mean()
