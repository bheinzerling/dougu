from collections.abc import Sequence, Iterable, Mapping

from functools import reduce
import six

from .decorators import cached_property


def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item


def split_lengths_for_ratios(nitems, *ratios):
    """Return the lengths of the splits obtained when splitting nitems
    by the given ratios"""
    lengths = [int(ratio * nitems) for ratio in ratios]
    i = 1
    while sum(lengths) != nitems and i < len(ratios):
        lengths[-i] += 1
        i += 1
    assert sum(lengths) == nitems, f'{sum(lengths)} != {nitems}\n{ratios}'
    return lengths


def split_idxs_for_ratios(nitems, *ratios, end_inclusive=False):
    import numpy as np
    assert len(ratios) >= 1
    assert all(0 < ratio < 1 for ratio in ratios)
    assert sum(ratios) <= 1.0
    idxs = list(np.cumsum(split_lengths_for_ratios(nitems, *ratios)))
    if end_inclusive:
        idxs = [0] + idxs
        if idxs[-1] != nitems:
            idxs.append(nitems)
    return idxs


def split_by_ratios(items, *ratios):
    nitems = len(items)
    split_idxs = split_idxs_for_ratios(nitems, *ratios, end_inclusive=True)
    return [
        items[split_idxs[i]:split_idxs[i+1]]
        for i in range(len(split_idxs) - 1)]


def split_by_lengths(items, *lengths):
    idxs = [0]
    for l in lengths:
        idxs.append(idxs[-1] + l)
    return [items[idxs[i]:idxs[i+1]] for i in range(len(idxs) - 1)]


def random_split_by_ratios(items, *ratios, inplace_shuffle=False):
    import random
    if not inplace_shuffle:
        items = list(items)
    random.shuffle(items)
    return split_by_ratios(items, *ratios)


def unordered_pairs(iterable):
    """Yield all unordered pairs of items in iterable.

    >>> list(unordered_pairs([1, 2, 3]))
    [(1, 2), (1, 3), (2, 3)]
    """
    from itertools import tee
    aa, bb = tee(iterable)
    try:
        next(bb)  # advance because we won't pair items with themselves
        for a in aa:
            bb, bb_copy = tee(bb)
            for b in bb:
                yield a, b
            bb = bb_copy
            next(bb)
    except StopIteration:
        return


def subsequences(items, subsequence_lengths):
    """Iterator over all contiguous subsequences of items with the given
    subsequence lengths."""
    for length in subsequence_lengths:
        for i in range(len(items) - length + 1):
            yield items[i:i + length]


def is_subseq(needle_seq, haystack_seq):
    """Determine if haystack_seq contains all items in needle_seq, in
    the same order."""
    # source: https://stackoverflow.com/questions/24017363/how-to-test-if-one-string-is-a-subsequence-of-another  # NOQA
    haystack_iter = iter(haystack_seq)
    return all(item in haystack_iter for item in needle_seq)


def to_from_idx(iterable, start_idx=0):
    """Return mappings of items in iterable to and from their index.

    >>> char2idx, idx2char = to_from_idx("abcdefg")
    >>> char2idx
    {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6}
    >>> idx2char
    {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g'}
    """
    return map(dict, zip(*(
        ((item, i), (i, item))
        for i, item in enumerate(iterable, start=start_idx))))


def map_assert(map_fn, assert_fn, iterable):
    """Assert that assert_fn is True for all results of applying
    map_fn to iterable"""
    for item in map(map_fn, iterable):
        assert assert_fn(item), item
        yield item


def map_skip_assert_error(map_fn, iterable, verbose=False):
    """Same as built-in map, but skip all items in iterabe that raise
    an assertion error when map_fn is appllied"""
    errors = 0
    for i, item in enumerate(iterable):
        try:
            yield map_fn(item)
        except AssertionError:
            if verbose:
                errors += 1
    if verbose:
        total = i + 1
        print(f"Skipped {errors} / {total} AssertionErrors")


def groupby(keys, values):
    """Group values according to their key."""
    d = {}
    for k, v in zip(keys, values):
        d.setdefault(k, []).append(v)
    return d


def groupby_lambda(key_fn, values):
    """Group values according to their key.
    The key is the result of applying key_fn to a value."""
    d = {}
    for v in values:
        k = key_fn(v)
        d.setdefault(k, []).append(v)
    return d


# https://stackoverflow.com/a/1055378
def is_non_string_iterable(arg):
    """Return True if arg is an iterable, but not a string."""
    return (
        isinstance(arg, Iterable)
        and not isinstance(arg, six.string_types)
    )


def to_list(item):
    """Puts item into a list if it isn't an iterable already."""
    if is_non_string_iterable(item):
        return item
    return [item]


class LazyList(Sequence):
    """Like list(generator), but with lazy evaluation, i.e.
    the generator is left unevaluated until first access."""
    def __init__(self, generator, name='items'):
        super().__init__()
        self.generator = generator
        self.name = name
        from .log import get_logger
        self.log = get_logger()

    @cached_property
    def items(self):
        items = list(self.generator)
        self.log.info(f'loaded {len(items)} {self.name}')
        return items

    def __getitem__(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.items)


def masked_select(items, mask):
    """Select items whose corresponding entry in mask is truthy.
    """
    return [item for item, entry in zip(items, mask) if entry]


def insert(_list, indexes_and_items):
    """Insert multiple items into _list.
    """
    for idx, item in sorted(indexes_and_items, reverse=True):
        _list.insert(idx, item)


def dict_argmax(d):
    """Returns the key with maximum associated value.
    """
    max_key = None
    max_value = float('-inf')
    for k, v in d.items():
        if v > max_value:
            max_value = v
            max_key = k
    return max_key


def transpose_dict(dict_of_lists):
    """'Transposes' a dictionary containing lists into a list of dictionaries.
    """
    keys = list(dict_of_lists.keys())
    values = list(dict_of_lists.values())
    for v in values:
        assert len(v) == len(values[0])
    return [dict(zip(keys, items)) for items in zip(*values)]


def concat_dict_values(dicts):
    """Aggrgates multiple dictionaries into a single dictionary by
    concatenating the corresponding values of each dictionary
    """
    d = {}
    for _dict in dicts:
        for k, v in _dict.items():
            d.setdefault(k, []).append(v)
    return d


def map_values(map_fn, dictionary):
    """Returns a dictionary whose values have been transformed by map_fn.
    """
    return {k: map_fn(v) for k, v in dictionary.items()}


def all_equal(items):
    """Returns True if all items are equal, else False.
    Unlike the typical len(set(items)) == 1 check, this function
    does not require that items are hashable.
    """
    sentinel = object()
    reduced = reduce(lambda a, b: a if a == b else sentinel, items)
    return reduced is not sentinel


def take_singleton(items):
    """Returns the first item in `items` if `items` contains exactly
    one item, otherwise raises an exception.
    """
    exactly_one_taken = False
    for item in items:
        if exactly_one_taken:
            exactly_one_taken = False
            break
        exactly_one_taken = True
    if not exactly_one_taken:
        raise ValueError('items does not contain exactly one item')
    return item


def index_select(items, indexes):
    """Select items by the specified indexes
    """
    return [items[idx] for idx in indexes]


class LazyDict(Mapping):
    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)
        self._cache = dict()

    def __getitem__(self, key):
        if key not in self._cache:
            func, arg = self._raw_dict.__getitem__(key)
            self._cache[key] = func(arg)
        return self._cache[key]

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)


class HashableDict(dict):
    def __init__(self, hashkey, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hashkey = hashkey

    @property
    def _key(self):
        return self[self.hashkey]

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return self._key == other._key
