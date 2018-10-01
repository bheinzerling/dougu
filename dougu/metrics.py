from pathlib import Path
from subprocess import run, PIPE
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix as cm
from boltons.iterutils import pairwise_iter as pairwise
from IPython import embed

from .iters import flatten
from .data import print_cm


CONLLEVAL = str((Path(__file__).parent / "../scripts/conlleval").absolute())


def f1_micro(true, pred):
    return f1_score(true, pred, average="micro")


def conll_ner(sents, pred, true, tag_enc=None, outfile=None, show_cm=False):
    if tag_enc is not None:
        pred = tag_enc.inverse_transform(pred)
        true = tag_enc.inverse_transform(true)
    token_lines = list(map(" ".join, zip(flatten(sents), true, pred)))
    sent_offsets = np.cumsum([0] + list(map(len, sents)))
    sent_lines = "\n\n".join(map(
        lambda p: "\n".join(token_lines[slice(*p)]), pairwise(sent_offsets)))
    if outfile:
        with outfile.open("w", encoding="utf8") as out:
            out.write(sent_lines)
    eval_out, eval_parsed = run_conll_eval(sent_lines)
    print(eval_out)
    if show_cm:
        try:
            print_cm(
                cm(true, pred, tag_enc.labels), tag_enc.labels, percent=True)
        except:
            pass
    return eval_parsed


def run_conll_eval(eval_in):
    out = run(CONLLEVAL, input=eval_in, encoding="utf8", stdout=PIPE).stdout
    return out, parse_conlleval(out)


def parse_conlleval(output):
    lines = output.replace("%", "").replace(" ", "").split("\n")[1:-1]
    return dict(map(
        lambda kv: (kv[0], float(kv[1])),
        map(lambda s: s.split(":"), lines[0].split(";"))))


class ConllScore():
    def __init__(self, tag_enc=None):
        self.sentences = []
        self.outfile = None
        self.tag_enc = tag_enc

    def __call__(self, pred, true):
        assert len(self.sentences) == len(pred) == len(true), embed()
        for s, p, t in zip(self.sentences, pred, true):
            assert len(s) == len(p) == len(t), embed()
        result = conll_ner(
            self.sentences, list(flatten(pred)), list(flatten(true)),
            tag_enc=self.tag_enc, outfile=self.outfile)
        return result["FB1"]
