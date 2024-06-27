import os
from pathlib import Path

from .misc import conf_hash, get_logger
from .io import mkdir


__all__ = [
    "cached_property",
    'file_cached_property',
    'torch_cached_property',
    'numpy_cached_property',
    'pandas_cached_property',
    'datasets_cached_property',
    'with_file_cache',
    'global_cached_property',
    ]


class _Missing(object):

    def __repr__(self):
        return 'no value'

    def __reduce__(self):
        return '_missing'


_missing = _Missing()

_global_cache = dict()


class _cached_property(property):
    """A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value:
        class Foo(object):
            @cached_property
            def foo(self):
                # calculate something important here
                return 42
    The class has to have a `__dict__` in order for this property to
    work.
    """
    # source: https://github.com/pallets/werkzeug/blob/master/werkzeug/utils.py

    # implementation detail: A subclass of python's builtin property
    # decorator, we override __get__ to check for a cached value. If one
    # choses to invoke __get__ by hand the property will still work as
    # expected because the lookup logic is replicated in __get__ for
    # manual invocation.

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __set__(self, obj, value):
        obj.__dict__[self.__name__] = value

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value

    def __delete__(self, obj):
        del obj.__dict__[self.__name__]

    # https://www.ianlewis.org/en/pickling-objects-cached-properties
    def __getstate__(self):
        state = self.__dict__.copy()
        for key in state:
            if (hasattr(self.__class__, key) and
                    isinstance(getattr(self.__class__, key), _cached_property)):
                pass
                # del state[key]
                # state[key] = self.func(key)
        return state


def cached_property(func=None, **kwargs):
    # https://stackoverflow.com/questions/7492068/python-class-decorator-arguments
    if func:
        return _cached_property(func)
    else:
        def wrapper(func):
            return _cached_property(func, **kwargs)

        return wrapper


class _file_cached_property(_cached_property):
    def __init__(
            self,
            func,
            loader=None,
            saver=None,
            cache_dir=Path('cache'),
            fname_tpl=None,
            reload_on_exception=False,
            **kwargs):
        super().__init__(func, **kwargs)
        self.cache_dir = cache_dir
        if loader is None:
            import joblib
            loader = joblib.load
        if saver is None:
            import joblib
            saver = joblib.dump
        self.loader = loader
        self.saver = saver
        self.func_name = func.__name__
        self.fname_tpl = fname_tpl
        self.reload_on_exception = reload_on_exception
        self.enabled = bool(int(os.environ.get('FILE_CACHE', '1')))
        self.log = get_logger().info

    def cache_file(self, obj):
        conf_str = getattr(obj, 'conf_str', 'no_conf')
        fname = (
            obj.__class__.__name__.lower() + '.' +
            self.func_name + '.' +
            (self.fname_tpl or
                '{conf_str}.pkl').format(conf_str=conf_str))
        mkdir(self.cache_dir)
        cache_file = self.cache_dir / fname
        return cache_file

    def __get__(self, obj, type=None):
        if not self.enabled:
            return self.func(obj)
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            cache_file = self.cache_file(obj)
            loaded = False
            if cache_file.exists():
                try:
                    self.log(f'loading {cache_file}')
                    value = self.loader(cache_file)
                    loaded = True
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    if self.reload_on_exception:
                        self.log('Exception while loading cache. Reloading')
                        loaded = False
                    else:
                        raise e
            if not loaded:
                self.log(f'saving {cache_file}')
                value = self.func(obj)
                self.saver(value, cache_file)
            obj.__dict__[self.__name__] = value
        return value


def file_cached_property(func=None, **kwargs):
    if func:
        return _file_cached_property(func)
    else:
        def wrapper(func):
            return _file_cached_property(func, **kwargs)
        return wrapper


class _global_cached_property(_file_cached_property):
    def key(self, obj):
        return str(self.cache_file(obj))

    def __set__(self, obj, value):
        _global_cache[self.key(obj)] = value

    def __get__(self, obj, type=None):
        key = self.key(obj)
        if obj is None:
            return self
        value = _global_cache.get(key, _missing)
        if value is _missing:
            value = super().__get__(obj, type=type)
            _global_cache[key] = value
        return value

    def __delete__(self, obj):
        del _global_cache[self.key(obj)]


def global_cached_property(func=None, **kwargs):
    if func:
        return _global_cached_property(func)
    else:
        def wrapper(func):
            return _global_cached_property(func, **kwargs)
        return wrapper


def torch_cached_property(
        func=None, map_location='cpu', fname_tpl='{conf_str}.pt', **kwargs):
    import torch

    def loader(f):
        return torch.load(f, map_location=map_location)

    if func:
        return _file_cached_property(
            func, loader=loader, saver=torch.save, fname_tpl=fname_tpl)
    else:
        def wrapper(func):
            return _file_cached_property(
                func,
                loader=loader,
                saver=torch.save,
                fname_tpl=fname_tpl,
                **kwargs)
        return wrapper


def numpy_cached_property(func=None, **kwargs):
    import numpy as np

    def saver(obj, cache_file):
        return np.save(cache_file, obj)

    if func:
        return _file_cached_property(func, loader=np.load, saver=saver)
    else:
        def wrapper(func):
            return _file_cached_property(
                func, loader=np.load, saver=saver, **kwargs)
        return wrapper


def hdf5_cached_property(func=None, **kwargs):
    import h5py

    def saver(obj, cache_file):
        assert hasattr(obj, 'shape')
        with h5py.File(cache_file, 'w') as f:
            dset = f.create_dataset('dataset', obj.shape, chunks=True)
            dset[:] = obj

    def loader(cache_file):
        with h5py.File(cache_file, 'r') as f:
            return f['dataset'][()]

    if func:
        return _file_cached_property(func, loader=loader, saver=saver)
    else:
        def wrapper(func):
            return _file_cached_property(
                func, loader=loader, saver=saver, **kwargs)
        return wrapper


def datasets_cached_property(func=None, **kwargs):
    from datasets import load_from_disk

    def saver(obj, cache_file):
        return obj.save_to_disk(cache_file)

    if func:
        return _file_cached_property(func, loader=load_from_disk, saver=saver)
    else:
        def wrapper(func):
            return _file_cached_property(
                func, loader=load_from_disk, saver=saver, **kwargs)
        return wrapper


def pandas_cached_property(func=None, **kwargs):
    def saver(obj, cache_file):
        return obj.to_pickle(cache_file)

    def loader(cache_file):
        import pandas as pd
        return pd.read_pickle(cache_file)

    if func:
        return _file_cached_property(func, loader=loader, saver=saver)
    else:
        def wrapper(func):
            return _file_cached_property(
                func, loader=loader, saver=saver, **kwargs)
        return wrapper


def with_file_cache(
        self,
        conf,
        *,
        loader=None,
        saver=None,
        fields=None,
        cache_dir=Path('cache'),
        fname_tpl=None,
        log=None,
        reload_on_exception=False):
    def actual_decorator(obj_loader):
        def wrapper(*args, **kwargs):
            c = conf
            if isinstance(c, str) and self is not None and hasattr(self, c):
                conf_str = getattr(self, c)()
            else:
                conf_str = conf_hash(c, fields)
            fname = (fname_tpl or '{conf_str}').format(conf_str=conf_str)
            cache_file = cache_dir / fname
            loaded = False
            if cache_file.exists():
                try:
                    if log:
                        log(f'loading {cache_file}')
                    nonlocal loader
                    if loader is None:
                        import joblib
                        loader = joblib.load
                    obj = loader(cache_file)
                    loaded = True
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    if reload_on_exception:
                        if log:
                            log('Exception while loading cache. Reloading')
                        loaded = False
                    else:
                        raise e
            if not loaded:
                obj = obj_loader(*args, **kwargs)
                if log:
                    log(f'saving {cache_file}')
                nonlocal saver
                if saver is None:
                    import joblib
                    saver = joblib.dump
                saver(obj, cache_file)
            if self is None:
                return obj
            else:
                for k, v in obj.items():
                    setattr(self, k, v)
        return wrapper
    return actual_decorator
