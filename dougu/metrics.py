from pathlib import Path
from subprocess import run, PIPE
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix as cm
from boltons.iterutils import pairwise_iter as pairwise

from .iters import flatten
from .data import print_cm


CONLLEVAL = str(Path(__file__).parent / "../scripts/conlleval")


def f1_micro(true, pred):
    return f1_score(true, pred, average="micro")


def conll_ner(docs, pred, true, label_enc=None, outfile=None):
    if label_enc is not None:
        pred = label_enc.inverse_transform(pred)
        true = label_enc.inverse_transform(true)
    token_lines = list(map(" ".join, zip(flatten(docs), pred, true)))
    doc_offsets = np.cumsum([0] + list(map(len, docs)))
    doc_lines = "\n\n".join(map(
        lambda p: "\n".join(token_lines[slice(*p)]), pairwise(doc_offsets)))
    if outfile:
        with outfile.open("w", encoding="utf8") as out:
            out.write(doc_lines)
    p = run(CONLLEVAL, input=doc_lines, encoding="utf8", stdout=PIPE)
    print(p.stdout)
    try:
        print_cm(
            cm(true, pred, label_enc.labels), label_enc.labels, percent=True)
    except:
        pass
    return parse_conlleval(p.stdout)


def parse_conlleval(output):
    lines = output.replace("%", "").replace(" ", "").split("\n")[1:-1]
    return dict(map(
        lambda kv: (kv[0], float(kv[1])),
        map(lambda s: s.split(":"), lines[0].split(";"))))
