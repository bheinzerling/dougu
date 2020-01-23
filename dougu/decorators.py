from pathlib import Path
from .misc import conf_hash


__all__ = ["cached_property", 'with_file_cache']


class _Missing(object):

    def __repr__(self):
        return 'no value'

    def __reduce__(self):
        return '_missing'


_missing = _Missing()


class cached_property(property):
    """A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value::
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


def with_file_cache(
        self,
        conf,
        *,
        loader,
        saver,
        fields=None,
        cache_dir=Path('cache'),
        cache_fname_tpl=None,
        log=None):
    def actual_decorator(data_dict_fn):
        def wrapper(*args, **kwargs):
            conf_str = conf_hash(conf, fields)
            cache_fname = (cache_fname_tpl or '{conf_str}').format(
                conf_str=conf_str)
            cache_file = cache_dir / cache_fname
            if cache_file.exists():
                if log:
                    log(f'loading {cache_file}')
                data_dict = loader(cache_file)
            else:
                data_dict = data_dict_fn(*args, **kwargs)
                if log:
                    log(f'saving {cache_file}')
                saver(data_dict, cache_file)
            for k, v in data_dict.items():
                setattr(k, v)
        return wrapper
    return actual_decorator
