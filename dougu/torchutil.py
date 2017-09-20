from pathlib import Path
from collections import defaultdict

import numpy as np
import torch


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
            return torch.from_numpy(args[0]).cuda()
        return FloatTensor
else:
    from torch import Tensor, LongTensor


def save_model(model, model_file, log=None):
    """Save a pytorch model to model_file"""
    if isinstance(model_file, str):
        model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("wb") as out:
        torch.save(model.state_dict(), out)
    if log:
        log.info("saved %s", model_file)


class LengthBatcher():
    def __init__(self, X, Y, batch_size, get_len=lambda x: x[1] - x[0]):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        len2idxs = defaultdict(list)
        for idx in range(len(X)):
            len2idxs[get_len(X[idx])].append(idx)
        self.len2idxs = {l: LongTensor(idxs) for l, idxs in len2idxs.items()}
        self.lengths = np.array(list(self.len2idxs.keys()))

    def __iter__(self):
        np.random.shuffle(self.lengths)
        for length in self.lengths:
            idxs = self.len2idxs[length]
            shuf_idxs = torch.randperm(idxs.shape[0])
            if torch.cuda.is_available():
                shuf_idxs.cuda()
            for batch_idxs in idxs[shuf_idxs].split(self.batch_size):
                yield self.X[batch_idxs], self.Y[batch_idxs]

    def print_stats(self):
        pprint({l: idxs.shape[0] for l, idxs in self.len2idxs.items()})
