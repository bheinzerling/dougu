import random
from argparse import ArgumentParser

from . import (
    conf_hash,
    add_jobid,
)


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
            for k, v in kwargs.items():
                if k in conf.__dict__:
                    conf.__dict__[k] = v
        self.conf = conf

    def arg_keys(self):
        return [
            arg[0][2:].replace('-', '_') for arg in getattr(self, 'args', [])
        ]

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
        a = AutoArgParser(description=desc)
        args = a.parse_args()
        add_jobid(args)
        return args


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

    def __init__(self):
        super().__init__()
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

    def __init__(self, conf, *args, random_seed=None, **kwargs):
        super().__init__(conf, *args, **kwargs)
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

    @property
    def random_state(self):
        return random.Random(self.random_seed)
