import json
from pathlib import Path


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
    with json_file.open("w", encoding="utf8") as out:
        json.dump(obj, out)


def jsonlines_load(jsonlines_file, max=None, skip=None):
    """Load objects from json lines file, i.e. a file with one
    serialized object per line."""
    yield from map(json.loads, lines(jsonlines_file, max=max, skip=skip))


def lines(file, max=None, skip=0):
    """Iterate over stripped lines in (text) file."""
    from itertools import islice
    try:
        from smart_open import smart_open as open
    except ImportError:
        pass
    with open(str(file), encoding="utf8") as f:
        for line in islice(f, skip, max):
            yield line.strip()


def dict_load(file, max=None, skip=0, splitter=None):
    """Load a dictionary from a text file containing one key-value
    pair per line."""
    if splitter is not None:
        if isinstance(splitter, (str, bytes)):
            def split(s): s.split(splitter)
        else:
            split = splitter
    else:
        split = str.split
    return dict(map(split, lines(file, max=max, skip=skip)))


def write_str(string, file, encoding="utf8"):
    """Write string to file."""
    try:
        from smart_open import smart_open as open
    except ImportError:
        pass
    with open(str(file), "w", encoding=encoding) as out:
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


def mkdir(dir, parents=True, exist_ok=True):
    """Convenience function for Path.mkdir"""
    dir = to_path(dir)
    dir.mkdir(parents=parents, exist_ok=exist_ok)
    return dir


def sentencepiece_load(file):
    """Load a SentencePiece model"""
    from sentencepiece import SentencePieceProcessor
    spm = SentencePieceProcessor()
    spm.Load(str(file))
    return spm
