import numpy as np


def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item


def split_idxs_for_ratios(nitems, *ratios, end_inclusive=False):
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


if __name__ == "__main__":
    print(split_idxs_for_ratios(100, 0.6, 0.2))
    for split in split_by_ratios(list(range(100)), 0.6, 0.2):
        print(split)

