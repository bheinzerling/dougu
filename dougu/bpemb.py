import re
from pathlib import Path

from .io import sentencepiece_load
from .embeddingutil import load_word2vec_file


class BPEmb():
    """
    Load a BPEmb model, preprocess text, encode/decode BPE using
    sentencepiece.
    """

    def __init__(
            self,
            *,
            lang="en", vs=10000, dim=100,
            model_file="data/bpemb/data/{lang}/{lang}.wiki.bpe.vs{vs}.model",
            bpemb_file="data/bpemb/data/{lang}/{lang}.wiki.bpe.vs{vs}.d{dim}.w2v.bin",  # NOQA
            spm_emb_idxs_match=True, preproc=True,
            encode_extra_options=None,
            add_pad=False,
            vs_fallback=True):
        if vs_fallback:
            available = BPEmb.available_vocab_sizes(lang, model_file)
            if not available:
                raise ValueError("No BPEmb models for " + model_file)
            if vs not in available:
                available = sorted(available)
                _vs = vs
                if vs < available[0]:
                    vs = available[0]
                else:
                    vs = available[-1]
                print(f"BPEmb fallback: {lang} from vocab size {_vs} to {vs}")
        self.lang = lang
        self.vocab_size = self.vs = vs
        self.dim = dim
        self.model_file = model_file.format(lang=lang, vs=vs)
        self.bpemb_file = bpemb_file.format(lang=lang, vs=vs, dim=dim)
        self.spm = sentencepiece_load(self.model_file)
        if encode_extra_options:
            self.spm.SetEncodeExtraOptions(encode_extra_options)
        self.emb = load_word2vec_file(self.bpemb_file, add_pad=add_pad)
        assert self.dim == self.emb.vectors.shape[1]
        self.preproc = preproc
        self.BOS = self.spm.PieceToId("<s>")
        self.EOS = self.spm.PieceToId("</s>")

    def __repr__(self):
        return self.__class__.__name__ + \
            f"(lang={self.lang}, vs={self.vocab_size}, dim={self.dim})"

    def encode_as_pieces(self, tokens):
        if self.preproc:
            tokens = self.do_preproc(tokens)
        return list(map(self.spm.EncodeAsPieces, tokens))

    def encode_as_ids(self, tokens):
        if self.preproc:
            tokens = self.do_preproc(tokens)
        return list(map(self.spm.EncodeAsIds, tokens))

    def encode_as_ids_with_eos(self, tokens):
        if self.preproc:
            tokens = self.do_preproc(tokens)
        return list(map(
            lambda t: self.spm.EncodeAsIds(t) + [self.EOS], tokens))

    def encode_as_ids_with_bos_eos(self, tokens):
        if self.preproc:
            tokens = self.do_preproc(tokens)
        return list(map(
            lambda t: [self.BOS] + self.spm.EncodeAsIds(t) + [self.EOS],
            tokens))

    def decode_ids(self, ids):
        try:
            return list(map(self.spm.DecodeIds, ids))
        except TypeError:
            try:
                return self.spm.DecodeIds(ids.tolist())
            except TypeError:
                return list(map(self.spm.DecodeIds, ids.tolist()))

    def do_preproc(self, tokens):
        return map(lambda t: re.sub("\d", "0", t.lower()), tokens)

    @property
    def pieces(self):
        return self.emb.index2word

    @property
    def words(self):
        return self.pieces

    @staticmethod
    def available_vocab_sizes(
            lang,
            model_file="data/bpemb/data/{lang}/{lang}.wiki.bpe.vs{vs}.model",
            vocab_sizes=[1000, 3000, 5000, 10000, 25000, 50000, 100000]):
        if lang.startswith("multi_"):
            vocab_sizes = [100000, 200000, 320000, 1000000]
        return set(
            vs for vs in vocab_sizes
            if Path(model_file.format(lang=lang, vs=vs)).exists())
