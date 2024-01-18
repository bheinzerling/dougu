import random
from argparse import ArgumentParser

from . import (
    conf_hash,
    add_jobid,
)
from .decorators import cached_property


class Configurable():
    classes = set()
    args = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Configurable.classes.add(cls)

    def __init__(self, conf=None, *args, **kwargs):
        super().__init__()
        if conf is None:
            conf = self.get_conf()
        if conf is not None and kwargs:
            from copy import deepcopy
            conf = deepcopy(conf)
        for k, v in kwargs.items():
            if k in conf.__dict__:
                conf.__dict__[k] = v
        self.conf = conf

    def arg_keys(self):
        return [
            arg[0][2:].replace('-', '_') for arg in getattr(self, 'args', [])
        ]

    @property
    def all_conf_fields(self):
        fields = self.conf_fields
        added = set(fields)
        for cls in self.__class__.__mro__:
            if cls is Configurable or cls is self.__class__:
                continue
            if issubclass(cls, Configurable):
                cls_conf_fields = super(cls, self).conf_fields
                for field in cls_conf_fields:
                    if field not in added:
                        added.add(field)
                        fields.append(field)
        return fields

    @property
    def conf_fields(self):
        return []


    @property
    def conf_str(self):
        return conf_hash(self.conf, self.conf_fields)

    def conf_str_for_fields(self, fields):
        return '.'.join([
            field + str(getattr(self.conf, field))
            for field in fields])

    @staticmethod
    def get_conf(desc='TODO'):
        a = Configurable.get_argparser(desc=desc)
        args = a.parse_args()
        add_jobid(args)
        return args

    @staticmethod
    def get_argparser(desc='TODO'):
        return AutoArgParser(description=desc)

    @staticmethod
    def parse_conf_dict(_dict):
        from types import SimpleNamespace
        a = Configurable.get_argparser()
        dest2action = {action.dest: action for action in a._get_optional_actions()}

        def parse_value(dest, value):
            if value is None:
                return value
            action = dest2action.get(dest, None)
            if action is None:
                return value
            ty = action.type
            if ty is None:
                return value
            if action.nargs in {'+', '*'}:
                return list(map(ty, value))
            return ty(value)

        return SimpleNamespace(**{
            dest: parse_value(dest, value)
            for dest, value in _dict.items()
            })


class AutoArgParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        added_names = dict()
        for cls in Configurable.classes:
            for arg in getattr(cls, 'args', []):
                name, kwargs = arg
                if name in added_names:
                    other_cls, other_kwargs = added_names[name]
                    if kwargs != other_kwargs:
                        raise ValueError(
                            f'Argument conflict. Argument "{name}" exists '
                            f'in {other_cls} with options {other_kwargs} '
                            f'and in {cls} with options {kwargs}')
                    else:
                        continue
                self.add_argument(name, **kwargs)
                added_names[name] = (cls, kwargs)


class EntryPoint(Configurable):
    args = Configurable.args + [
        ('command', dict(type=str, nargs='?')),
    ]

    def run(self):
        getattr(self, self.conf.command)()


class WithRandomSeed(Configurable):
    args = Configurable.args + [
        ('--random-seed', dict(type=int, default=2)),
    ]

    @property
    def conf_fields(self):
        return super().conf_fields + [
            'random_seed',
        ]

    def __init__(self, *args, random_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        if random_seed is None:
            random_seed = self.conf.random_seed
        self.random_seed = random_seed
        self.set_random_seed(random_seed)

    def set_random_seed(self, seed):
        import numpy as np
        import torch

        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class WithRandomState(WithRandomSeed):

    def set_random_seed(self, seed):
        # do not set any global random seed
        pass

    @cached_property
    def random_state(self):
        return random.Random(self.random_seed)

    @cached_property
    def numpy_random_state(self):
        from numpy.random import RandomState
        return RandomState(self.random_seed)

    @cached_property
    def numpy_rng(self):
        import numpy.random
        return numpy.random.default_rng(self.random_seed)

    @cached_property
    def pytorch_random_state(self):
        import torch
        rng = torch.Generator()
        rng.manual_seed(self.random_seed)
        return rng

    def sample(self, items, sample_size):
        import torch
        rng = self.pytorch_random_state
        rnd_idxs = torch.randperm(len(items), generator=rng)[:sample_size]
        if isinstance(items, torch.Tensor):
            sample = items[rnd_idxs]
        else:
            sample = list(map(items.__getitem__, rnd_idxs))
        assert len(sample) == sample_size
        return sample
