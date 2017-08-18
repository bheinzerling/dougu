from pathlib import Path

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
