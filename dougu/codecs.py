import numpy as np
import torch
import dougu.torchutil


class NgramCodec(object):
    """Encodes and decodes text into n-gram sequences"""
    def __init__(
            self,
            order,
            vocab_size,
            left_pad_symbol="<^>",
            right_pad_symbol="<$>",
            unk_symbol="<unk>",
            log=None):

        self.order = order
        self.vocab_size = vocab_size
        self.unk_symbol = unk_symbol
        self.special_symbols = []
        if left_pad_symbol:
            self.special_symbols.append(left_pad_symbol)
            self.left_pad = [left_pad_symbol]
        if right_pad_symbol:
            self.special_symbols.append(right_pad_symbol)
            self.right_pad = [right_pad_symbol]
        if unk_symbol:
            self.special_symbols.append(unk_symbol)
        self.pad_len_left = 1 if left_pad_symbol else 0
        self.pad_len_right = 1 if right_pad_symbol else 0
        self.vocab_size = vocab_size + len(self.special_symbols)
        self.log = log
        if self.log:
            self.log.info(
                "ngram order: %s. ngram vocab size: %s. special symbols %s",
                order, vocab_size, self.special_symbols)

    def fit(self, strings):
        from collections import Counter
        if isinstance(strings, str):
            if self.log:
                if len(strings) < 100:
                    self.log.warn(
                        "Fitting a single short string. Is this intended?")
            strings = [strings]
        o = self.order
        ngrams = [
            "".join(s[i:i + o])
            for s in map(self._pad, strings)
            for i in range(len(s) - o + 1)]

        n_most_common = self.vocab_size + len(self.special_symbols)
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

    def _pad(self, s):
        return self.left_pad + list(s) + self.right_pad

    def transform(self, strings):
        if isinstance(strings, str):
            strings = [strings]
        o = self.order
        unk = self.ngram2idx[self.unk_symbol]
        transformed = []
        for s in strings:
            s = self._pad(s)
            idxs = [
                self.ngram2idx.get("".join(s[i:i + o]), unk)
                for i in range(len(s) - o + 1)]
            transformed.append(idxs)
        return transformed

    def fit_transform(self, strings):
        self.fit(strings)
        return self.transform(strings)

    def inverse_transform(self, idxss):
        return [[self.idx2ngram[idx] for idx in idxs] for idxs in idxss]


class MultiNgramCodec(object):
    """Encodes and decodes text into 1-gram, 2-gram, ..., order-gram
    sequences"""
    def __init__(
            self, orders=[1, 2, 3],
            vocab_sizes=[100, 1000, 10000],
            left_pad_symbol="<^>",
            right_pad_symbol="<$>",
            unk_symbol="<unk>"):
        assert len(orders) == len(vocab_sizes)
        self.order2codec = {
            order: NgramCodec(
                order, vocab_size,
                left_pad_symbol, right_pad_symbol, unk_symbol)
            for order, vocab_size in zip(orders, vocab_sizes)}

    def fit(self, strings):
        for codec in self.order2codec.values():
            codec.fit(strings)
        return self

    def transform(self, strings):
        return {
            order: codec.transform(strings)
            for order, codec in self.order2codec.items()}

    def inverse_transform(self, order2idxss):
        return {
            order: self.order2codec[order].inverse_transform(idxss)
            for order, idxss in order2idxss.items()}


class DictLabelEncoder():
    def fit(self, labels):
        self.label2idx = {label: idx for idx, label in enumerate(labels)}
        self.idx2label = self.classes_ = labels
        return self

    def transform(self, labels):
        return np.array(list(map(self.label2idx.__getitem__, labels)))

    def inverse_transform(self, idxs):
        return list(map(self.idx2label.__getitem__, idxs))


class LabelEncoder(object):
    """Encodes and decodes labels. Decoding from idx representation.
    Optionally return pytorch tensors instead of numpy arrays.
    Use backend 'sklearn' (default) for speed or backend 'dict' if the
    order of labels should be preserved (i.e. first label will be idx 0...)."""
    def __init__(self, to_torch=False, device="cuda", backend='sklearn'):
        self.to_torch = to_torch
        self.device = device
        if backend == 'sklearn':
            from sklearn.preprocessing import LabelEncoder as _LabelEncoder
            self._LabelEncoder = _LabelEncoder
        elif backend == 'dict':
            self._LabelEncoder = DictLabelEncoder
        else:
            raise ValueError('unknown backend:', backend)

    def fit(self, labels, min_count=0, unk_label=None):
        self.unk_label = unk_label
        if min_count > 0:
            from collections import Counter
            counts = Counter(labels)
            labels = [
                label for label in set(labels)
                if counts[label] > min_count]
            assert unk_label is not None
            labels.append(unk_label)
            self.label_set = set(labels)
        self.label_enc = self._LabelEncoder().fit(labels)
        self.labels = self.label_enc.classes_
        self.nlabels = len(self.labels)
        idxs = list(range(self.nlabels))
        self.idx2label = dict(zip(idxs, self.inverse_transform(idxs)))
        return self

    def __len__(self):
        return self.nlabels

    def transform(self, labels):
        if isinstance(labels, str):
            labels = [labels]
        if labels and isinstance(labels[0], list):
            return [self.transform(l) for l in labels]
        if self.unk_label is not None:
            labels = [
                label if label in self.label_set else self.unk_label
                for label in labels]
        if self.to_torch:
            if not labels:
                return torch.tensor([]).to(
                    dtype=torch.int64, device=self.device)
            tensors = []
            bs = 1000000
            for i in range(0, len(labels), bs):
                labels_enc = self.label_enc.transform(labels[i:i+bs])
                tensors.append(torch.LongTensor(labels_enc))
            return torch.cat(tensors).to(device=self.device)
        else:
            return self.label_enc.transform(labels)

    def inverse_transform(self, idx, ignore_idx=None):
        if ignore_idx is not None:
            def filter_idxs(idxs):
                return [i for i in idxs if i != ignore_idx]
        else:
            def filter_idxs(idxs):
                return idxs
        try:
            idx = idx.tolist()
        except AttributeError:
            pass
        try:
            if isinstance(idx[0], list):
                return [
                    self.label_enc.inverse_transform(
                        filter_idxs(_idx)).tolist()
                    for _idx in idx]
        except TypeError:
            return self.label_enc.inverse_transform([idx])[0]
        return self.label_enc.inverse_transform(filter_idxs(idx))

    @staticmethod
    def from_file(
            file,
            additional_labels=None,
            to_torch=False,
            save_to=None,
            device="cuda"):
        """Create LabelEncoder instance from file, which contains
        one label per line. Optionally dump instance to save_to."""
        from .io import lines
        codec = LabelEncoder(to_torch, device=device)
        if additional_labels is None:
            additional_labels = []
        codec.fit(list(lines(file)) + additional_labels)
        if save_to:
            import joblib
            joblib.dump(codec, save_to)
        return codec


class LabelOneHotEncoder(object):
    """Encodes and decodes labels. Decoding either from idx or one-hot
    representation. Optionally return pytorch tensors instead of numpy
    arrays."""
    def __init__(self, to_torch=False, reduce=None):
        self.to_torch = to_torch
        self.reduce = reduce

    def fit(self, labels):
        from sklearn.preprocessing import (
            LabelEncoder as _LabelEncoder, LabelBinarizer)
        self.label_enc = _LabelEncoder().fit(labels)
        labels_enc = self.label_enc.transform(labels)
        self.one_hot_enc = LabelBinarizer().fit(labels_enc)
        self.nlabels = len(self.label_enc.classes_)
        return self

    def transform_idx(self, labels):
        if isinstance(labels, str):
            labels = [labels]
        labels_enc = self.label_enc.transform(labels)
        if self.to_torch:
            return torch.LongTensor(labels_enc)
            # return torch.from_numpy(labels_enc).long().cuda()
        return labels_enc

    def transform_one_hot(self, labels):
        if isinstance(labels, str):
            labels = [labels]
        labels_enc = self.label_enc.transform(labels)
        if self.to_torch:
            t = self.one_hot_enc.transform(labels_enc)
            # return Tensor(t.astype(float)).long()
            one_hot = torch.LongTensor(t)
            if self.reduce == 'sum':
                return one_hot.sum(dim=0)
            return one_hot
        one_hot = np.array(self.one_hot_enc.transform(labels_enc))
        if self.reduce == 'sum':
            return one_hot.sum(axis=0)
        return one_hot

    def inverse_transform_one_hot(self, one_hot):
        idx = self.one_hot_enc.inverse_transform(one_hot)
        return self.label_enc.inverse_transform(idx)

    def inverse_transform_idx(self, idx):
        return self.label_enc.inverse_transform(idx)


if __name__ == "__main__":
    strings = ["abcde", "test", "string"]
    codec = MultiNgramCodec(orders=[1, 2, 3], vocab_sizes=[10, 100, 400])
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
