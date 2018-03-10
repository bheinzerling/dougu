import json
import logging
from pathlib import Path
from itertools import islice

from smart_open import smart_open


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def to_path(maybe_str):
    if isinstance(maybe_str, str):
        return Path(maybe_str)
    return maybe_str


def json_load(json_file):
    """Load object from json file."""
    with to_path(json_file).open(encoding="utf8") as f:
        return json.load(f)


def json_dump(obj, json_file):
    """Dump obj to json file."""
    with smart_open(json_file, "w", encoding="utf8") as out:
        json.dump(obj, out)


def jsonlines_load(jsonlines_file, max=None):
    """Load objects from json lines file, i.e. a file with one
    serialized object per line."""
    yield from map(json.loads, lines(jsonlines_file, max))


def lines(file, max=None):
    """Iterate over stripped lines in (text) file."""
    with smart_open(str(file), encoding="utf8") as f:
        for line in islice(f, 0, max):
            yield line.strip()


def write_str(string, file, encoding="utf8"):
    """Write string to file."""
    with smart_open(str(file), "w", encoding=encoding) as out:
        out.write(string)


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
    with to_path(file).open("w", encoding="utf8") as out:
        json.dump({k: str(v) for k, v in args.__dict__.items()}, out, indent=4)


def sentencepiece_load(file):
    from sentencepiece import SentencePieceProcessor
    spm = SentencePieceProcessor()
    spm.Load(str(file))
    return spm
