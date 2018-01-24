from pathlib import Path
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.cuda import LongTensor


# fix inconsistencies between cuda and non-cuda tensors when
# constructing from numpy arrays:
# "tried to construct a tensor from a int sequence, but found an item of type numpy.int64"  # NOQA
if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor as _LongTensor

    def LongTensor(*args, **kwargs):
        if isinstance(args[0], np.ndarray):
            return torch.from_numpy(args[0]).cuda()
        return _LongTensor(*args, **kwargs)

    def Tensor(*args, **kwargs):
        if isinstance(args[0], np.ndarray):
            return torch.from_numpy(args[0]).cuda().float()
        return FloatTensor
else:
    from torch import Tensor, LongTensor  # NOQA

# if torch.cuda.is_available():
#     from torch.cuda import FloatTensor, LongTensor
#     Tensor = FloatTensor
# else:
#     from torch import Tensor, LongTensor  # NOQA
#     FloatTensor = Tensor


class LengthBatcher():
    def __init__(
            self, X, Y=None, batch_size=100,
            get_len=lambda x: x[1] - x[0], keys=None,
            start_ends=False):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        if start_ends:
            keys = X[:, 1] - X[:, 0]
        if keys is None:
            len2idxs = defaultdict(list)
            for idx in range(len(X)):
                len2idxs[get_len(X[idx])].append(idx)
            self.len2idxs = {l: LongTensor(idxs) for l, idxs in len2idxs.items()}
            self.lengths = np.array(list(self.len2idxs.keys()))
            self.multilen = self.lengths.ndim > 1
        else:
            self.lengths = list(set(keys.cpu().tolist()))
            self.len2idxs = {
                l: torch.nonzero(keys == l).squeeze()
                for l in self.lengths}
            self.multilen = False

    def __iter__(self):
        if torch.cuda.is_available():
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
        else:
            if self.Y is not None:
                yield from self.iter_XY_cpu()
            else:
                yield from self.iter_X_cpu()

    def __len__(self):
        return sum(
            len(idxs.split(self.batch_size))
            for idxs in self.len2idxs.values())

    def iter_XY(self):
        np.random.shuffle(self.lengths)
        for length in self.lengths:
            idxs = self.len2idxs[length]
            shuf_idxs = torch.randperm(idxs.shape[0]).cuda()
            for batch_idxs in idxs[shuf_idxs].split(self.batch_size):
                yield self.X[batch_idxs], self.Y[batch_idxs]

    def iter_X(self):
        np.random.shuffle(self.lengths)
        for length in self.lengths:
            idxs = self.len2idxs[length]
            shuf_idxs = torch.randperm(idxs.shape[0]).cuda()
            for batch_idxs in idxs[shuf_idxs].split(self.batch_size):
                yield self.X[batch_idxs]

    def iter_X_multi(self):
        np.random.shuffle(self.lengths)
        for lengths in self.lengths:
            idxs = self.len2idxs[tuple(lengths)]
            shuf_idxs = torch.randperm(idxs.shape[0]).cuda()
            for batch_idxs in idxs[shuf_idxs].split(self.batch_size):
                yield self.X[batch_idxs]

    def iter_XY_cpu(self):
        np.random.shuffle(self.lengths)
        for length in self.lengths:
            idxs = self.len2idxs[length]
            shuf_idxs = torch.randperm(idxs.shape[0])
            for batch_idxs in idxs[shuf_idxs].split(self.batch_size):
                yield self.X[batch_idxs], self.Y[batch_idxs]

    def iter_X_cpu(self):
        np.random.shuffle(self.lengths)
        for length in self.lengths:
            idxs = self.len2idxs[length]
            shuf_idxs = torch.randperm(idxs.shape[0])
            for batch_idxs in idxs[shuf_idxs].split(self.batch_size):
                yield self.X[batch_idxs]

    def print_stats(self):
        pprint({l: idxs.shape[0] for l, idxs in self.len2idxs.items()})


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


def emb_layer(keyed_vectors, backprop=False):
    emb_weights = Tensor(keyed_vectors.syn0)
    emb = nn.Embedding(*emb_weights.shape)
    emb.weight = nn.Parameter(emb_weights)
    emb.weight.requires_grad = backprop
    return emb
