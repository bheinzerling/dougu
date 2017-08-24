import json
import logging
from pathlib import Path
from itertools import islice


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def to_path(maybe_str):
    if isinstance(maybe_str, str):
        return Path(maybe_str)
    return maybe_str


def json_load(json_file):
    """Load object from json file."""
    json_file = to_path(json_file)
    with json_file.open(encoding="utf8") as f:
        return json.load(f)


def json_dump(obj, json_file):
    """Dump obj to json file."""
    json_file = to_path(json_file)
    with json_file.open("w", encoding="utf8") as out:
        json.dump(obj, out)


def jsonlines_load(jsonlines_file, max=None):
    """Load objects from json lines file, i.e. a file with one
    serialized object per line."""
    yield from map(json.loads, lines(jsonlines_file, max))


def load_word2vec_file(word2vec_file, weights_file=None, normalize=False):
    """Load a word2vec file in either text or bin format, optionally
    supplying custom embedding weights and normalizing embeddings."""
    from gensim.models import KeyedVectors
    binary = word2vec_file.endswith(".bin")
    log.info("loading %s", word2vec_file)
    vecs = KeyedVectors.load_word2vec_format(word2vec_file, binary=binary)
    if weights_file:
        import torch
        weights = torch.load(weights_file)
        vecs.syn0 = weights.cpu().float().numpy()
    if normalize:
        log.info("normalizing %s", word2vec_file)
        vecs.init_sims(replace=True)
    return vecs


def lines(file, max=None):
    """Iterate over stripped lines in (text) file."""
    file = to_path(file)
    with file.open(encoding="utf8") as f:
        for line in islice(f, 0, max):
            yield line.strip()


def deserialize_protobuf_instances(cls, protobuf_file, max_bytes=None):
    """Deserialze a protobuf file into instances of cls"""
    from google.protobuf.internal.decoder import _DecodeVarint32
    with open(protobuf_file, "rb") as f:
        buf = f.read(max_bytes)
    n = 0
    while n < len(buf):
        msg_len, new_pos = _DecodeVarint32(buf, n)
        n = new_pos
        msg_buf = buf[n:n+msg_len]
        n += msg_len
        c = cls()
        c.ParseFromString(msg_buf)
        yield c


def dump_args(args, file):
    """Write argparse args to file."""
    with file.open("w", encoding="utf8") as out:
        json.dump({k: str(v) for k, v in args.__dict__.items()}, out, indent=4)
