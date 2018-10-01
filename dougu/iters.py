def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item


def split_idxs_for_ratios(nitems, *ratios, end_inclusive=False):
    import numpy as np
    assert len(ratios) >= 1
    assert all(0 < ratio < 1 for ratio in ratios)
    assert sum(ratios) <= 1.0
    idxs = list(np.cumsum([int(ratio * nitems) for ratio in ratios]))
    if end_inclusive:
        return [0] + idxs + [nitems]
    return idxs


def split_by_ratios(items, *ratios):
    nitems = len(items)
    split_idxs = split_idxs_for_ratios(nitems, *ratios, end_inclusive=True)
    return [
        items[split_idxs[i]:split_idxs[i+1]]
        for i in range(len(split_idxs) - 1)]


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
    d = {}
    for k, v in zip(keys, values):
        d.setdefault(k, []).append(v)
    return d


if __name__ == "__main__":
    print(split_idxs_for_ratios(100, 0.6, 0.2))
    for split in split_by_ratios(list(range(100)), 0.6, 0.2):
        print(split)
