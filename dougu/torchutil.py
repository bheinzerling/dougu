from pathlib import Path
from collections import defaultdict
from pprint import pprint
import random

import numpy as np
import torch
from torch import nn, optim, tensor


class TensorBatcher():
    def __init__(
            self, X, Y=None, *, batch_size=64, shuffle=True):
        self.X = X
        if Y is not None:
            assert X.size(0) == Y.size(0)
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle

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
    def __init__(
            self, X, Y=None, batch_size=100,
            get_len=lambda x: x[1] - x[0], keys=None,
            start_ends=False, log=None):
        self.X = X
        self.Y = Y
        self.device = self.X.device
        if self.Y is not None:
            assert self.X.device == self.Y.device
        self.batch_size = batch_size
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
        else:
            self.lengths = list(set(keys.cpu().tolist()))
            self.len2idxs = {
                l: torch.nonzero(keys == l).squeeze()
                for l in self.lengths}
            self.multilen = False
        if log:
            log.info(f"{len(self)} batches. batch size: {self.batch_size}")

    def __iter__(self):
        if self.Y is not None:
            if self.multilen:
                raise NotImplementedError
            else:
                yield from self.iter_XY()
        else:
            if self.multilen:
                yield from self.iter_X_multi()
            else:
                yield from self.iter_X()

    def __len__(self):
        return sum(
            len(idxs.split(self.batch_size))
            for idxs in self.len2idxs.values())

    def iter_XY(self):
        np.random.shuffle(self.lengths)
        for length in self.lengths:
            idxs = self.len2idxs[length]
            shuf_idxs = torch.randperm(idxs.shape[0]).to(self.device)
            for batch_idxs in idxs[shuf_idxs].split(self.batch_size):
                yield self.X[batch_idxs], self.Y[batch_idxs]

    def iter_X(self):
        np.random.shuffle(self.lengths)
        for length in self.lengths:
            idxs = self.len2idxs[length]
            shuf_idxs = torch.randperm(idxs.shape[0]).to(self.device)
            for batch_idxs in idxs[shuf_idxs].split(self.batch_size):
                yield self.X[batch_idxs]

    def iter_X_multi(self):
        np.random.shuffle(self.lengths)
        for lengths in self.lengths:
            idxs = self.len2idxs[tuple(lengths)]
            shuf_idxs = torch.randperm(idxs.shape[0]).to(self.device)
            for batch_idxs in idxs[shuf_idxs].split(self.batch_size):
                yield self.X[batch_idxs]

    def print_stats(self):
        pprint(self.stats)

    @property
    def stats(self):
        return {l: idxs.shape[0] for l, idxs in self.len2idxs.items()}


def save_model(model, model_file, log=None):
    """Save a pytorch model to model_file"""
    if isinstance(model_file, str):
        model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("wb") as out:
        torch.save(model.state_dict(), out)
    if log:
        log.info("saved %s", model_file)


def load_model(model, model_file):
    model.load_state_dict(torch.load(model_file))


def emb_layer(keyed_vectors, trainable=False, use_weights=True, **kwargs):
    emb_weights = tensor(keyed_vectors.syn0)
    emb = nn.Embedding(*emb_weights.shape, **kwargs)
    if use_weights:
        emb.weight = nn.Parameter(emb_weights)
    emb.weight.requires_grad = trainable
    return emb


class Score():
    """Keep track of a score computed by score_func, save model
    if score improves.
    """
    def __init__(self, name, score_func, shuffle_baseline=False):
        self.name = name
        self.current = 0.0
        self.best = 0.0
        self.best_model = None
        self.pred = []
        self.true = []
        self.shuffle = []
        self.score_func = score_func
        self.shuffle_baseline = shuffle_baseline

    def extend(self, pred, true):
        """append predicted and true labels"""
        self.pred.extend(pred)
        self.true.extend(true)

    def update(self, model=None, rundir=None, epoch=None):
        score = self.score_func(self.true, self.pred)
        self.current_score = score
        if score > self.best:
            self.best = score
            if model:
                assert rundir
                epoch_str = f"e{epoch}_" if epoch is not None else ""
                fname = f"{epoch_str}{self.name}_{score:.4f}_model.pt"
                model_file = rundir / fname
                save_model(model, model_file)
                self.best_model = model_file
        if self.shuffle_baseline:
            random.shuffle(self.pred)
            shuffle_score = self.score_func(self.true, self.pred)
        else:
            shuffle_score = None
        self.true = []
        self.pred = []
        return score, shuffle_score

    def update_log(self, model=None, rundir=None, epoch=None, log=None):
        score, shuffle_score = self.update(
            model=model, rundir=rundir, epoch=epoch)
        s = f"score {self.name}_{score:.4f}/{self.best:.4f}\n{self.best_model}"
        if shuffle_score is not None:
            s += f"\nshuffle {self.name}_{shuffle_score:.4f}"
        if log:
            log.info(s)
        else:
            print(s)

    @property
    def best_str(self):
        return f"{self.name}_{self.best:.4f}"

    @property
    def current_str(self):
        return f"{self.name}_{self.current_score:.4f}"


class LossTracker(list):
    def __init__(self, name):
        self.name = name
        self.best_loss = defaultdict(lambda: float("inf"))
        self.best_model = None

    def interval_end(
            self, epoch=None, model=None, model_file=None, ds_name=None):
        loss = np.average(self)
        print(loss, self.best_loss[ds_name])
        if loss < self.best_loss[ds_name]:
            self.best_loss[ds_name] = loss
            if model:
                model_file = Path(str(model_file).format(
                    epoch=epoch,
                    ds_name=ds_name,
                    loss=loss))
                save_model(model, model_file)
                self.best_model = model_file
        self.clear()
        return loss


class LossTrackers():
    def __init__(self, *loss_trackers):
        self.loss_trackers = loss_trackers

    def append(self, *losses):
        for lt, loss in zip(self.loss_trackers, losses):
            lt.append(loss.item())

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
        print(f"e{epoch} {ds_name} " + " ".join(
            f"{name}_{loss:.4f}/{best:.4f}"
            for name, loss, best in self.interval_end(
                epoch=epoch,
                model=model, model_file=model_file, ds_name=ds_name)))

    def best_log(self):
        print("best: " + " ".join(
            f"{lt.name}_{lt.best_loss:.6f}" for lt in self.loss_trackers))

    @staticmethod
    def from_names(*names):
        return LossTrackers(*map(LossTracker, names))

    def __iter__(self):
        return iter(self.loss_trackers)

    def __getitem__(self, i):
        return self.loss_trackers[i]


def get_optim(args, model):
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim.lower() == "adam":
        return optim.Adam(params, lr=args.learning_rate)
    elif args.optim.lower() == "sgd":
        return optim.SGD(params, lr=args.learning_rate, momentum=args.momentum)
    raise ValueError("Unknown optimizer: " + args.optim)


def scalar(tensor):
    maybe_scalar = tensor.cpu().numpy()
    assert maybe_scalar.size == 1
    return maybe_scalar[0]
