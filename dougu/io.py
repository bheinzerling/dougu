import json
from pathlib import Path
from typing import IO


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
        out.write("\n")


def jsonlines_load(jsonlines_file, max=None, skip=None, filter_fn=None):
    """Load objects from json lines file, i.e. a file with one
    serialized object per line."""
    if filter_fn is not None:
        yielded = 0
        for line in lines(jsonlines_file, skip=skip):
            obj = json.loads(line)
            if filter_fn(obj):
                yield obj
                yielded += 1
                if max and yielded >= max:
                    break
    else:
        yield from map(json.loads, lines(jsonlines_file, max=max, skip=skip))


def lines(file, max=None, skip=0, apply_func=str.strip):
    """Iterate over lines in (text) file. Optionally skip first `skip` 
    lines, only read the first `max` lines, and apply `apply_func` to 
    each line. By default lines are stripped, set `apply_func` to None 
    to disable this."""
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
        max=None, skip=0, splitter=None,
        key_index=0, value_index=1,
        key_apply=None, value_apply=None):
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
            return key_apply(parts[key_index]), value_apply(parts[value_index])
    elif key_apply is not None:
        def kv(line):
            parts = split(line)
            return key_apply(parts[key_index]), parts[value_index]
    elif value_apply is not None:
        def kv(line):
            parts = split(line)
            return parts[key_index], value_apply(parts[value_index])
    else:
        def kv(line):
            parts = split(line)
            return parts[key_index], parts[value_index]
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


def _maybe_to_str(v):
    try:
        json.dumps(v)
    except TypeError:
        return str(v)
    return v


def args_to_json(args):
    """Same as json.dumps, but more lenient by converting non-serializable
    objects like PosixPaths to strings."""
    return json.dumps({k: _maybe_to_str(v) for k, v in args.__dict__.items()})


def dump_args(args, file):
    """Write argparse args to file."""
    with to_path(file).open("w", encoding="utf8") as out:
        json.dump({
            k: _maybe_to_str(v)
            for k, v in args.__dict__.items()}, out, indent=4)


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


def load_obj(path, obj_getter):
    """Load dumped object from path. If the object has not been dumped
    before, create it by invoking obj_getter, then dump it."""
    try:
        import joblib
        load = joblib.load
        dump = joblib.dump
    except ImportError:
        import pickle
        load = pickle.load
        dump = pickle.dump
    try:
        obj = load(path)
    except FileNotFoundError:
        obj = obj_getter()
        dump(obj, path)
    return obj


# source: https://github.com/allenai/allennlp/blob/master/allennlp/common/file_utils.py#L147  # NOQA
def http_get_temp(url: str, temp_file: IO) -> None:
    import requests
    from tqdm import tqdm
    req = requests.get(url, stream=True)
    req.raise_for_status()
    content_length = req.headers.get('Content-Length')
    print(req.headers)
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()
    return req.headers


# source: https://github.com/allenai/allennlp/blob/master/allennlp/common/file_utils.py#L147  # NOQA
def http_get(url: str, outfile: Path) -> None:
    import tempfile
    import shutil
    from .log import get_logger
    log = get_logger()
    with tempfile.NamedTemporaryFile() as temp_file:
        headers = http_get_temp(url, temp_file)
        # we are copying the file before closing it, flush to avoid truncation
        temp_file.flush()
        # shutil.copyfileobj() starts at current position, so go to the start
        temp_file.seek(0)
        mkdir(outfile.parent)
        if headers.get("Content-Type") == "application/x-gzip":
            import tarfile
            tf = tarfile.open(fileobj=temp_file)
            members = tf.getmembers()
            if len(members) != 1:
                raise NotImplementedError("TODO: extract multiple files")
            member = members[0]
            tf.extract(member, outfile.parent)
            assert (outfile.parent / member.name) == outfile
        else:
            with open(outfile, 'wb') as out:
                shutil.copyfileobj(temp_file, out)
    return outfile
