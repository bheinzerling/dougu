from pathlib import Path
from collections import defaultdict, deque
from pprint import pprint
import random
import heapq

import sklearn
import numpy as np
import torch
from torch import nn, optim, tensor, arange
from torch.utils.data import (
    RandomSampler, BatchSampler, Dataset, random_split, DataLoader)

from .iters import flatten, split_lengths_for_ratios, split_by_ratios


class TensorBatcher():
    """Simple combination of a TensorDataset and a DataLoader, but
    faster due to less overhead."""
    def __init__(
            self, X, Y=None, *, batch_size=64, shuffle=True):
        self.X = X
        if Y is not None:
            assert X.size(0) == Y.size(0)
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        print("X", X.shape, len(self), "batches")

    def __iter__(self):
        if self.shuffle:
            get_idxs = torch.randperm
        else:
            get_idxs = torch.arange
        b_idxs = get_idxs(self.X.size(0)).to(self.X.device)
        if self.Y is not None:
            assert self.X.device == self.Y.device
            yield from zip(
                self.X[b_idxs].split(self.batch_size),
                self.Y[b_idxs].split(self.batch_size))
        else:
            yield from self.X[b_idxs].split(self.batch_size)

    def __len__(self):
        return (self.X.size(0) - 1) // self.batch_size + 1


class LengthBatcher():
    """Splits variable-length instances in X and Y into batches,
    according to their length. Instances in a batch have the same length,
    as determined by get_len. Instances in different batches may have
    different lengths."""

    def __init__(
            self, X, Y=None, batch_size=100,
            get_len=lambda x: (x[1] - x[0]).item(), keys=None,
            start_ends=False, log=None, iter_idxs=False, shuffle=True,
            lengths_on_cpu=False):
        self.X = X
        self.Y = Y
        self.device = self.X.device
        if self.Y is not None:
            assert self.X.device == self.Y.device
        self.batch_size = batch_size
        self.iter_idxs = iter_idxs
        self.shuffle = shuffle
        if start_ends:
            keys = X[:, 1] - X[:, 0]
        if keys is None:
            len2idxs = defaultdict(list)
            for idx in range(len(X)):
                len2idxs[get_len(X[idx])].append(idx)
            self.len2idxs = {
                l: tensor(idxs, dtype=torch.int64, device=self.device)
                for l, idxs in len2idxs.items()}
            self.lengths = np.array(list(self.len2idxs.keys()))
            self.multilen = self.lengths.ndim > 1
            if self.multilen:
                self.lengths = [tuple(lengths) for lengths in self.lengths]
        else:
            if lengths_on_cpu:
                keys = keys.cpu()
            self.lengths = list(set(keys.cpu().tolist()))
            self.len2idxs = {
                l: torch.nonzero(keys == l).squeeze(dim=-1)
                for l in self.lengths}
            self.multilen = False
        if log:
            log.info(f"{len(self)} batches. batch size: {self.batch_size}")

    def __iter__(self):
        batches = []
        for length in self.lengths:
            idxs = self.len2idxs[length]
            if self.shuffle:
                idxs = idxs[torch.randperm(idxs.shape[0]).to(self.device)]
            for batch_idxs in idxs.split(self.batch_size):
                batches.append(batch_idxs)
        if self.shuffle:
            random.shuffle(batches)

        if self.iter_idxs:
            yield from batches
        elif self.Y is not None:
            if self.multilen:
                raise NotImplementedError
            else:
                for batch_idxs in batches:
                    yield self.X[batch_idxs], self.Y[batch_idxs]
        else:
            for batch_idxs in batches:
                yield self.X[batch_idxs]

    def __len__(self):
        return sum(
            len(idxs.split(self.batch_size))
            for idxs in self.len2idxs.values())

    def print_stats(self):
        pprint(self.stats)

    @property
    def stats(self):
        return {l: idxs.shape[0] for l, idxs in self.len2idxs.items()}


def save_model(model, model_file, log=None):
    """Save a pytorch model to model_file."""
    if isinstance(model_file, str):
        model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("wb") as out:
        torch.save(model.state_dict(), out)
    if log:
        log.info("saved %s", model_file)


class Checkpoints():
    """Creates and keeps track of model checkpoints, automatically
    deleting old ones. Deletion is based either on a checkpoint 'priority'
    score (e.g. model accuracy on a dev set) or 'recency'."""
    def __init__(
            self, checkpoint_dir,
            max_checkpoints=10, queue_mode="priority", priority_mode="max"):
        self.max_checkpoints = max_checkpoints
        if priority_mode == "max":
            self.priority_factor = 1
        elif priority_mode == "min":
            self.priority_factor = -1
        else:
            raise ValueError("Unknown priority_mode: " + priority_mode)
        if queue_mode == "priority":
            self.checkpoints = []
            heapq.heapify(self.checkpoints)
            self._append = self._append_heap
        elif queue_mode == "recency":
            self.checkpoints = deque(maxlen=self.max_checkpoints)
            self._append = self._append_deque

    def _append_heap(self, priority, checkpoint_file):
        priority *= self.priority_mode
        heapq.heappush(self.checkpoints, (priority, checkpoint_file))
        if len(self.checkpoints) > self.max_checkpoints:
            return heapq.heappop(self.checkpoints)

    def _append_deque(self, checkpoint_file):
        if len(self.checkpoints) == self.max_checkpoints:
            oldest = self.checkpoints.popleft()
        else:
            oldest = None
        self.checkpoints.append(checkpoint_file)
        return oldest

    def append(self, *args):
        to_delete = self._append(*args)
        if to_delete:
            to_delete.unlink()


def load_model(model, model_file, strict=True):
    """Load model weights from model_file."""
    model.load_state_dict(torch.load(model_file), strict=True)


def emb_layer(
        vecs, trainable=False, use_weights=True, dtype=torch.float32,
        device='cuda',
        **kwargs):
    """Create an Embedding layer from a gensim KeyedVectors instance
     or an embedding matrix."""
    try:
        emb_weights = tensor(vecs.syn0, dtype=dtype).to(device=device)
    except AttributeError:
        emb_weights = vecs
    emb = nn.Embedding(*emb_weights.shape, **kwargs)
    if use_weights:
        emb.weight = nn.Parameter(emb_weights)
    emb.weight.requires_grad = trainable
    return emb


class Score():
    """Keep track of a score computed by score_func, save model
    if score improves.
    """
    def __init__(
            self, name, score_func=None, shuffle_baseline=False,
            comp=float.__gt__, save_model=True, log=None,
            add_mode="extend"):
        self.name = name
        if comp == float.__lt__:
            self.current = float("inf")
            self.best = float("inf")
            self.optimum = "min"
        else:
            self.current = 0.0
            self.best = 0.0
            self.optimum = "max"
        self.best_model = None
        self.pred = []
        self.true = []
        self.shuffle = []
        self.score_func = score_func or self.accuracy
        self.shuffle_baseline = shuffle_baseline
        self.comp = comp
        self.save_model = save_model
        self.info = log.info if log else print
        if add_mode == "extend":
            self.add = self.extend
        elif add_mode == "append":
            self.add = self.append
        else:
            raise ValueError("Unknown add_mode: " + add_mode)

    def extend(self, pred, true=None):
        """extend predicted and true labels"""
        if hasattr(pred, "tolist"):
            pred = pred.tolist()
        self.pred.extend(pred)
        if true is not None:
            if hasattr(pred, "shape"):
                assert pred.shape == true.shape, (pred.shape, true.shape)
            else:
                assert len(pred) == len(true)
            if hasattr(true, "tolist"):
                true = true.tolist()
            self.true.extend(true)

    def append(self, pred, true=None):
        """append predicted and true labels"""
        if hasattr(pred, "tolist"):
            pred = pred.tolist()
        self.pred.append(pred)
        if true is not None:
            if hasattr(pred, "shape"):
                assert pred.shape == true.shape, (pred.shape, true.shape)
            else:
                assert len(pred) == len(true)
            if hasattr(true, "tolist"):
                true = true.tolist()
            self.true.append(true)

    def update(self, model=None, rundir=None, epoch=None, score=None):
        if score is None:
            if not self.true:
                score = self.score_func(self.pred)
            else:
                score = self.score_func(self.pred, self.true)
        self.current = score
        if self.comp(score, self.best):
            self.best = score
            if self.save_model and model:
                assert rundir
                epoch_str = f"e{epoch}_" if epoch is not None else ""
                fname = f"{epoch_str}{self.name}_{score:.4f}_model.pt"
                model_file = rundir / fname
                save_model(model, model_file)
                self.best_model = model_file
        if self.shuffle_baseline:
            random.shuffle(self.pred)
            shuffle_score = self.score_func(self.pred, self.true)
        else:
            shuffle_score = None
        self.true = []
        self.pred = []
        return score, shuffle_score

    def update_log(
            self, model=None, rundir=None, epoch=None, score=None):
        score, shuffle_score = self.update(
            model=model, rundir=rundir, epoch=epoch, score=score)
        self.info(f"score {self.name}_{score:.4f}/{self.best:.4f}")
        if self.best_model:
            self.info(str(self.best_model))
        if shuffle_score is not None:
            self.info(f"\nshuffle {self.name}_{shuffle_score:.4f}")
        return score

    @staticmethod
    def accuracy(pred, true):
        n = len(pred)
        assert n != 0
        assert n == len(true)
        correct = sum(p == t for p, t in zip(pred, true))
        return correct / n

    @staticmethod
    def f1_score(pred, true):
        return sklearn.metrics.f1_score(true, pred)

    @staticmethod
    def f1_score_multiclass(pred, true, average='macro'):
        f1_score = sklearn.metrics.f1_score
        if average == 'macro':
            return np.average([f1_score(t, p) for t, p in zip(true, pred)])
        elif average == 'micro':
            return f1_score(list(flatten(true)), list(flatten(pred)))

    @staticmethod
    def f1_score_multiclass_micro(pred, true):
        return Score.f1_score_multiclass(pred, true, average='micro')

    @staticmethod
    def f1_score_multiclass_macro(pred, true):
        return Score.f1_score_multiclass(pred, true, average='macro')

    @property
    def best_str(self):
        return f"{self.name}_{self.best:.4f}"

    @property
    def current_str(self):
        return f"{self.name}_{self.current_score:.4f}"


class LossTracker(list):
    """Keep track of losses, save model if loss improves."""
    def __init__(self, name, save_model=True, log=None):
        self.name = name
        self.best_loss = defaultdict(lambda: float("inf"))
        self.best_model = None
        self.save_model = save_model
        self.info = log.info if log else print

    def interval_end(
            self, epoch=None, model=None, model_file=None, ds_name=None):
        loss = self.current
        if loss < self.best_loss[ds_name]:
            self.best_loss[ds_name] = loss
            if self.save_model and model:
                model_file = Path(str(model_file).format(
                    epoch=epoch,
                    ds_name=ds_name,
                    loss=loss))
                save_model(model, model_file)
                self.best_model = model_file
        self.clear()
        return loss

    @property
    def current(self):
        return np.average(self)


class LossTrackers():
    """Keep track of multiple losses."""
    def __init__(self, *loss_trackers, log=None):
        self.loss_trackers = loss_trackers
        self.info = log.info if log else print

    def append(self, *losses):
        for lt, loss in zip(self.loss_trackers, losses):
            try:
                loss = loss.item()
            except AttributeError:
                pass
            lt.append(loss)

    def interval_end(
            self, *, epoch=None, model=None, model_file=None, ds_name=None):
        for lt in self.loss_trackers:
            yield (
                lt.name,
                lt.interval_end(
                    epoch=epoch,
                    model=model, model_file=model_file, ds_name=ds_name),
                lt.best_loss[ds_name])

    def interval_end_log(
            self, epoch, *, model=None, model_file=None, ds_name=None):
        self.info(f"e{epoch} {ds_name} " + " ".join(
            f"{name}_{loss:.6f}/{best:.6f}"
            for name, loss, best in self.interval_end(
                epoch=epoch,
                model=model, model_file=model_file, ds_name=ds_name)))

    def best_log(self):
        self.info("best: " + " ".join(
            f"{lt.name}_{lt.best_loss:.6f}" for lt in self.loss_trackers))

    @staticmethod
    def from_names(*names, **kwargs):
        loss_trackers = map(lambda name: LossTracker(name, **kwargs), names)
        return LossTrackers(*loss_trackers, log=kwargs.get("log"))

    def __iter__(self):
        return iter(self.loss_trackers)

    def __getitem__(self, i):
        return self.loss_trackers[i]


def get_optim(
        conf, model, optimum='max', n_train_instances=None,
        additional_params_dict=None):
    """Create an optimizer according to command line args."""
    params = [p for p in model.parameters() if p.requires_grad]
    optim_name = conf.optim.lower()
    if optim_name == "adam":
        return optim.Adam(params, lr=conf.learning_rate)
    elif optim_name == "adamw":
        from transformers import AdamW
        no_decay = ['bias', 'LayerNorm.weight']
        additional_params = (
            set(additional_params_dict['params'])
            if additional_params_dict
            else {})
        grouped_params = [
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not p in additional_params],
                'weight_decay': conf.weight_decay},
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not p in additional_params],
                'weight_decay': 0.0}]
        if additional_params_dict:
            grouped_params.append(additional_params_dict)
        return AdamW(grouped_params, lr=conf.learning_rate, eps=1e-8)
    elif optim_name == "sgd":
        return optim.SGD(
            params,
            lr=conf.learning_rate,
            momentum=conf.momentum,
            weight_decay=conf.weight_decay)
    elif optim_name == 'bertadam':
        return get_bert_optim(conf, model, n_train_instances)
    elif optim_name == 'radam':
        from .radam import RAdam
        return RAdam(params, lr=conf.learning_rate)
    raise ValueError("Unknown optimizer: " + conf.optim)


def get_lr_scheduler(conf, optimizer, optimum='max'):
    if conf.learning_rate_scheduler == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        lr_scheduler = ReduceLROnPlateau(
            optimizer, factor=0.5,
            patience=conf.learning_rate_scheduler_patience,
            mode=optimum,
            verbose=True)
    elif conf.learning_rate_scheduler:
        raise ValueError(
            "Unknown lr_scheduler: " + conf.learning_rate_scheduler)
    else:
        lr_scheduler = None
    return lr_scheduler


def get_optim_and_lr_scheduler(
        conf, model, optimum='max', n_train_instances=None):
    optimizer = get_optim(conf, model, n_train_instances=n_train_instances)
    lr_scheduler = get_lr_scheduler(conf, optimizer, optimum=optimum)
    return optimizer, lr_scheduler


def get_bert_optim(args, model, n_train_instances):
    from pytorch_pretrained_bert.optimization import BertAdam
    num_train_optimization_steps = int(
            n_train_instances /
            args.batch_size /
            args.gradient_accumulation_steps) * args.max_epochs
    if args.bert_half_precision:
        param_optimizer = [
            (n, param.clone().detach().to('cpu').float().requires_grad_())
            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [
            (n, param.clone().detach().to('cpu').requires_grad_())
            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in param_optimizer
            if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01}, {
        'params': [
            p for n, p in param_optimizer
            if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}]
    t_total = num_train_optimization_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    return BertAdam(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        warmup=args.warmup_proportion,
        t_total=t_total)


def tensorize_varlen_items(
        items,
        device="cuda",
        item_dtype=torch.int64,
        startends_dtype=torch.int64):
    """Tensorize variable-length items, e.g. a list of sentences in,
    a document with each sentence being a list of word indexes.
    This is done by creating a 'store' vector which contains the items
    in sequential order (e.g., word indexes as they occur in the document),
    and a 'startends' tensor which contains the start and end offsets of
    each item (e.g. the start and end offset of each sentence).
    """
    store = torch.tensor(list(flatten(items)), device=device, dtype=item_dtype)
    lengths = list(map(len, items))
    starts = np.cumsum([0] + lengths[:-1])
    ends = np.cumsum(lengths)
    startends = np.stack([starts, ends]).T
    startends = torch.tensor(startends, device=device, dtype=startends_dtype)
    return store, startends


def take_from_store(startends, store):
    """Create a batch of token indices corresponding to the start and end
    offsets in the batch of same-length startends.

    Arguments:

    startends (Tensor with Shape(batch_size x 2)):
        A batch of start and end offsets encoding sequences of the same length.
    store (Tensor with Shape(n_tokens)):
        A token store created by tensorize_varlen_items
    """
    batch_size = startends.size(0)
    seq_length = startends[0, 1] - startends[0, 0]
    offsets = arange(seq_length).to(device=store.device).repeat(batch_size, 1)
    idx = startends[:, 0:1].expand_as(offsets) + offsets
    return store.take(idx)


# source: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping():
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ListDataset(Dataset):
    def __init__(self, instances, max_instances=None):
        super().__init__()
        if max_instances is not None and max_instances < len(instances):
            instances = [
                instances[idx]
                for idx in torch.randperm(max_instances)]
        self.instances = instances

    def __getitem__(self, index):
        return self.instances.__getitem__(index)

    def __len__(self):
        return self.instances.__len__()


class TensorDataset(Dataset):
    """Same as pytorch's TensorDataset, but doesn't require tensors to
    have a .size method"""
    def __init__(self, *tensors):
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class TransposedTensorDataset(Dataset):
    """Same as pytorch's TensorDataset, but instead of yielding
    batch_size n-tuples, yields n tensors of len batch_size.
    Use with batch_sampler in DataLoader.
    """
    def __init__(self, *tensors):
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, batch_idxs):
        return [tensor[batch_idxs] for tensor in self.tensors]

    def __len__(self):
        return self.tensors[0].size(0)

    def batch_sampler(self, batch_size):
        idxs = torch.arange(len(self))
        return BatchSampler(
            RandomSampler(idxs), batch_size=batch_size, drop_last=False)


def batch_sampler(dataset, batch_size):
    idxs = torch.arange(len(dataset))
    return BatchSampler(
        RandomSampler(idxs), batch_size=batch_size, drop_last=False)


class Splits():
    def __init__(
            self,
            dataset,
            split_ratios=(0.8, 0.1, 0.1),
            split_lengths=None,
            split_names=('train', 'dev', 'test'),
            splits=None):
        self.split_names = split_names
        self.split_ratios = split_ratios
        self.split_lengths = split_lengths or (
            split_lengths_for_ratios(len(dataset), *split_ratios))
        if splits is None:
            splits = self._split(dataset)
        for name, split in zip(split_names, splits):
            setattr(self, name, split)

    def _split(self, dataset):
        return split_by_ratios(dataset, self.split_ratios)

    def loaders(self, *args, split_names=None, **kwargs):
        if not split_names:
            split_names = self.split_names
        return {
            split_name: getattr(
                self, split_name + '_loader')(*args, **kwargs)
            for split_name in split_names}

    def train_loader(self, *args, **kwargs):
        assert 'train' in self.split_names
        return DataLoader(self.train, *args, **kwargs)

    def dev_loader(self, *args, **kwargs):
        assert 'dev' in self.split_names
        return DataLoader(self.dev, *args, **kwargs, shuffle=False)

    def test_loader(self, *args, **kwargs):
        assert 'test' in self.split_names
        return DataLoader(self.test, *args, **kwargs, shuffle=False)


class RandomSplits(Splits):
    def _split(self, instances):
        if sum(self.split_lengths) < len(instances):
            rnd_idxs = iter(torch.randperm(len(instances)))
            splits = [
                [instances[next(rnd_idxs)] for _ in range(split_length)]
                for split_length in self.split_lengths]
        else:
            splits = random_split(instances, self.split_lengths)
        if hasattr(instances, 'batch_sampler'):
            breakpoint()
        return splits


RandomSplitDataset = RandomSplits
