from collections import defaultdict
import random
from pprint import pprint

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels


class BatchedByLength():
    """Batch Xy by length, optionally using the get_len function
    to determine the length of X instances."""
    def __init__(
            self,
            Xy,
            batch_size,
            keep_leftovers=False,
            get_len=len,
            return_X_tensors=False,
            return_y_tensors=False,
            batch_first=False,
            shuffle=False
            ):

        if keep_leftovers:
            raise NotImplementedError("TODO: keep_leftovers")
        self.Xy = Xy
        self.batch_size = batch_size
        if return_X_tensors or return_y_tensors:
            import torch
            self.stack = torch.stack
        self.return_X_tensors = return_X_tensors
        self.return_y_tensors = return_y_tensors
        self.batch_first = batch_first
        self.shuffle = shuffle

        self.len2Xy = defaultdict(list)
        for inst in self.Xy:
            X = inst[0]
            self.len2Xy[get_len(X)].append(inst)
        self.Xy_batched = []
        if keep_leftovers:
            self.Xy_leftovers = []
        for l in sorted(self.len2Xy.keys()):
            Xy_l = self.len2Xy[l]
            # crop to fit batch_size
            too_many = len(Xy_l) % batch_size
            Xy_l = Xy_l[:-too_many or None]
            self.Xy_batched.extend(Xy_l)
            if keep_leftovers:
                Xy_leftover = Xy_l[-too_many:]
                if Xy_leftover:
                    self.Xy_leftovers.extend(Xy_leftover)

    def __getitem__(self, index):
        index = index * self.batch_size
        batch = self.Xy_batched[index:index + self.batch_size]
        try:
            X_batch, y_batch, *rest = zip(*batch)
        except:
            raise IndexError
        if self.return_X_tensors:
            X_batch = self.stack(X_batch, 1)
            if self.batch_first:
                X_batch = X_batch.transpose(0, 1)
        if self.return_y_tensors:
            y_batch = self.stack(y_batch, dim=1)
            if self.batch_first:
                y_batch = y_batch.transpose(0, 1)
        if rest:
            return X_batch, y_batch, rest
        return X_batch, y_batch

    def __len__(self):
        assert len(self.Xy_batched) % self.batch_size == 0
        n_batches = int(len(self.Xy_batched) / self.batch_size) - 1
        return n_batches

    def __iter__(self):
        idxs = range(len(self))
        if self.shuffle:
            idxs = list(idxs)
            random.shuffle(idxs)
        for i in idxs:
            yield self[i]

    @property
    def stats(self):
        return {
            l: len(self.len2Xy[l]) for l in sorted(self.len2Xy.keys())}


# https://github.com/scikit-learn/scikit-learn/blob/ab93d65/sklearn/metrics/classification.py  # NOQA
# slightly modified to report weighted, macro, and micro averages in one go.
def classification_report(y_true, y_pred, labels=None, target_names=None,
                          sample_weight=None, digits=2):
    """Build a text report showing the main classification metrics
    Read more in the :ref:`User Guide <classification_report>`.
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.
    target_names : list of strings
        Optional display names matching the labels (same order).
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    digits : int
        Number of digits for formatting output floating point values
    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score for each class.
    Examples
    --------
    >>> from sklearn.metrics import classification_report
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(classification_report(y_true, y_pred, target_names=target_names))
                 precision    recall  f1-score   support
    <BLANKLINE>
        class 0       0.50      1.00      0.67         1
        class 1       0.00      0.00      0.00         1
        class 2       1.00      0.67      0.80         3
    <BLANKLINE>
    avg / total       0.70      0.60      0.61         5
    <BLANKLINE>
    """

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    last_line_heading = 'avg / total'

    if target_names is None:
        target_names = ['%s' % l for l in labels]
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight)

    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ["{0}".format(s[i])]
        report += fmt % tuple(values)

    report += '\n'
    values = ["weighted " + last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average="macro",
                                                  sample_weight=sample_weight)

    # compute averages
    values = ["macro " + last_line_heading]
    for v in (p, r, f1):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average="micro",
                                                  sample_weight=sample_weight)

    # compute averages
    values = ["micro " + last_line_heading]
    for v in (p, r, f1):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)
    return report


def print_cm(
        cm, labels,
        percent=False,
        hide_zeroes=True, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    total = cm.sum(axis=1)
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            if percent:
                cell = "%{0}.1f".format(columnwidth) % (
                    cm[i, j] / total[i] * 100)
            else:
                cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


# https://github.com/glample/tagger/blob/master/utils.py
def ensure_iob2(tags):
    """Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True
