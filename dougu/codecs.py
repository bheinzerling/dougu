import logging
from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import torch
from torch.cuda import FloatTensor as Tensor


__all__ = [
    "NgramCodec",
    "MultiNgramCodec",
    "LabelOneHotEncoder"]


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class NgramCodec(object):
    """Encodes and decodes text into n-gram sequences"""
    def __init__(
            self,
            order,
            vocab_size,
            pad_symbol="<pad>",
            unk_symbol="<unk>"):

        self.order = order
        self.vocab_size = vocab_size
        self.pad_symbol = pad_symbol
        self.unk_symbol = unk_symbol
        self.special_symbols = []
        if pad_symbol is not None:
            self.special_symbols.append(pad_symbol)
        if unk_symbol is not None:
            self.special_symbols.append(unk_symbol)
        self.vocab_size = vocab_size + len(self.special_symbols)
        log.info(
            "ngram order: %s. ngram vocab size: %s. special symbols %s",
            order, vocab_size, self.special_symbols)

    def fit(self, strings):
        if isinstance(strings, str):
            if len(strings) < 100:
                log.warn("Fitting a single short string. Is this intended?")
            strings = [strings]
        o = self.order
        ngrams = [
            s[i:i + o]
            for s in strings
            for i, _ in enumerate(s)]

        n_most_common = self.vocab_size - len(self.special_symbols)
        most_common_ngrams = [
            ngram
            for ngram, count
            in Counter(ngrams).most_common(n_most_common)]
        self.classes_ = self.special_symbols + most_common_ngrams
        self.ngram2idx = {
            ngram: idx for idx, ngram in enumerate(self.classes_)}
        self.idx2ngram = {
            idx: ngram for idx, ngram in enumerate(self.classes_)}
        return self

    def transform(self, strings, pad_left=False, pad_right=False):
        if isinstance(strings, str):
            strings = [strings]
        if self.pad_symbol:
            pad = [self.pad_symbol] * self.order
        else:
            pad = []
        pad_len_left = len(pad) * int(pad_left)
        pad_len_right = len(pad) * int(pad_right)
        o = self.order
        unk = self.ngram2idx[self.unk_symbol]
        transformed = []
        left_padding = pad if pad_left else []
        right_padding = pad if pad_right else []
        for s in strings:
            s = left_padding + list(s) + right_padding
            last_char = len(s) - pad_len_right
            idxs = [
                self.ngram2idx.get("".join(s[i:i + o]), unk)
                for i, _ in enumerate(s[pad_len_left:last_char])]
            transformed.append(idxs)
        return transformed

    def fit_transform(self, strings):
        self.fit(strings)
        return self.transform(strings)

    def inverse_transform(self, idxss):
        return [
            "".join([self.idx2ngram[idx] for idx in idxs])
            for idxs in idxss]


class MultiNgramCodec(object):
    """Encodes and decodes text into 1-gram, 2-gram, ..., order-gram
    sequences"""
    def __init__(
            self, orders=[1, 2, 3],
            vocab_sizes=[100, 1000, 10000],
            pad_symbol="<pad>",
            unk_symbol="<unk>"):
        assert len(orders) == len(vocab_sizes)
        self.order2codec = {
            order: NgramCodec(order, vocab_size, pad_symbol, unk_symbol)
            for order, vocab_size in zip(orders, vocab_sizes)}

    def fit(self, strings):
        for codec in self.order2codec.values():
            codec.fit(strings)

    def transform(self, strings, pad_left=False, pad_right=True):
        return {
            order: codec.transform(strings, pad_left, pad_right)
            for order, codec in self.order2codec.items()}

    def inverse_transform(self, order2idxss):
        return {
            order: self.order2codec[order].inverse_transform(idxss)
            for order, idxss in order2idxss.items()}


class LabelOneHotEncoder(object):
    """Encodes and decodes labels. Decoding either from idx or one-hot
    representation. Optionally return pytorch tensors instead of numpy
    arrays."""
    def __init__(self, to_torch=False):
        self.to_torch = to_torch

    def fit(self, labels):
        self.label_enc = LabelEncoder().fit(labels)
        labels_enc = self.label_enc.transform(labels)
        self.one_hot_enc = LabelBinarizer().fit(labels_enc)
        self.nlabels = len(labels)
        return self

    def transform_idx(self, labels):
        if isinstance(labels, str):
            labels = [labels]
        labels_enc = self.label_enc.transform(labels)
        if self.to_torch:
            return torch.from_numpy(labels_enc).long().cuda()
        return labels_enc

    def transform_one_hot(self, labels):
        if isinstance(labels, str):
            labels = [labels]
        labels_enc = self.label_enc.transform(labels)
        if self.to_torch:
            t = self.one_hot_enc.transform(labels_enc)
            return Tensor(t.astype(float)).long()
        return np.array(self.one_hot_enc.transform(labels_enc))

    def inverse_transform_one_hot(self, one_hot):
        idx = self.one_hot_enc.inverse_transform(one_hot)
        return self.label_enc.inverse_transform(idx)

    def inverse_transform_idx(self, idx):
        return self.label_enc.inverse_transform(idx)


if __name__ == "__main__":
    strings = ["abcde", "test", "string"]
    codec = MultiNgramCodec(orders=[1, 2, 3], vocab_sizes=[5, 20, 40])
    codec.fit(strings)
    enc = codec.transform(strings)
    for idxs in enc.values():
        print(idxs)
    for ngrams in codec.inverse_transform(enc).values():
        print(ngrams)
    labels = ["A", "B", "C"]
    label_enc = LabelOneHotEncoder().fit(labels)
    labels_enc = label_enc.transform_idx(labels)
    print(labels_enc)
    labels_dec = label_enc.inverse_transform_idx(labels_enc)
    print(labels_dec)
