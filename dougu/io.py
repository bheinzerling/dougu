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
    with to_path(json_file).open("w", encoding="utf8") as out:
        json.dump(obj, out)


def jsonlines_load(jsonlines_file, max=None, skip=None):
    """Load objects from json lines file, i.e. a file with one
    serialized object per line."""
    yield from map(json.loads, lines(jsonlines_file, max=max, skip=skip))


def lines(file, max=None, skip=0, apply_func=str.strip):
    """Iterate over stripped lines in (text) file."""
    from itertools import islice
    if apply_func:
        with open(str(file), encoding="utf8") as f:
            for line in islice(f, skip, max):
                yield apply_func(line)
    else:
        with open(str(file), encoding="utf8") as f:
            for line in islice(f, skip, max):
                yield line


def dict_load(
        file,
        max=None, skip=0, splitter=None, key_apply=None, value_apply=None):
    """Load a dictionary from a text file containing one key-value
    pair per line."""
    if splitter is not None:
        if isinstance(splitter, (str, bytes)):
            def split(s):
                return s.split(splitter)
        else:
            split = splitter
    else:
        split = str.split
    if key_apply is not None and value_apply is not None:
        def kv(line):
            parts = split(line)
            return key_apply(parts[0]), value_apply(parts[1])
    elif key_apply is not None:
        def kv(line):
            parts = split(line)
            return key_apply(parts[0]), parts[1]
    elif value_apply is not None:
        def kv(line):
            parts = split(line)
            return parts[0], value_apply(parts[1])
    else:
        kv = split
    return dict(map(kv, lines(file, max=max, skip=skip)))


def write_str(string, file, encoding="utf8"):
    """Write string to file."""
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


# https://stackoverflow.com/a/27077437
def cat(infiles, outfile, buffer_size=1024 * 1024 * 100):
    """Concatenate infiles and write result to outfile, like Linux cat."""
    import shutil

    with outfile.open("wb") as out:
        for infile in infiles:
            with infile.open("rb") as f:
                shutil.copyfileobj(f, out, buffer_size)
