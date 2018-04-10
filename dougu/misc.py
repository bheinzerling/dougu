from pathlib import Path
from bisect import bisect


def now_str():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def autocommit(
        repodir=".", glob_pattern="**/*.py", recursive=True, msg="autocommit"):
    """Commit all changes in repodir in files matching glob_pattern."""
    from git import Repo
    import glob
    repo = Repo(repodir)
    files = glob.glob(glob_pattern, recursive=recursive)
    repo.index.add(files)
    if repo.index.diff(repo.head.commit):
        repo.index.commit(msg)
    sha = repo.head.object.hexsha
    return sha


class Results():
    """Thin wrapper around a pandas DataFrame to keep track of
    experimental results."""

    def __init__(self, file):
        import joblib
        from pandas import DataFrame as DF
        self.joblib = joblib
        self.DF = DF
        self.fname_pkl = Path(str(file) + ".df.pkl")
        self.fname_txt = Path(str(file) + ".txt")
        self.fname_stats = Path(str(file) + ".overall_stats")
        self.results = DF()
        if self.fname_pkl.exists():
            try:
                self.results = joblib.load(self.fname_pkl)
            except EOFError:
                import traceback
                traceback.print_exc()

    def append(self, results_row):
        row = self.DF([results_row])
        if self.results is None:
            self.results = row
        else:
            self.results = self.results.append(row, ignore_index=True)
        self.joblib.dump(self.results, self.fname_pkl)
        with self.fname_txt.open("a", encoding="utf8") as out:
            out.write("\t".join(list(map(str, results_row.values()))) + "\n")
        with self.fname_stats.open("w", encoding="utf8") as out:
            out.write(str(self.results.describe()))

    def max(self, score_fun=lambda df: df.mean(1)):
        return float(score_fun(self.results).max())


def get_and_increment_runid(file=Path("runid")):
    try:
        with file.open() as f:
            runid = int(f.read()) + 1
    except FileNotFoundError:
        runid = 0
    with file.open("w") as out:
        out.write(str(runid))
    return runid


def next_rundir(basedir, runid_fn="runid", log=None):
    runid = get_and_increment_runid(basedir / runid_fn)
    rundir = basedir / str(runid)
    rundir.mkdir(exist_ok=True, parents=True)
    if log:
        log.info(f"rundir: {rundir.resolve()}")
    return rundir


def color_range(start_color, end_color, steps=10, cformat=lambda c: c.hex_l):
    from colour import Color
    if isinstance(start_color, str):
        start_color = Color(start_color)
    if isinstance(end_color, str):
        end_color = Color(end_color)
    return list(map(cformat, start_color.range_to(end_color, steps)))


def random_string(length=8, chars=None):
    if chars is None:
        import string
        chars = string.ascii_letters + string.digits
    import random
    return "".join(random.choices(chars, k=length))


class Spans():
    """Find span covering a given offset. Assumes non-overlapping spans"""
    def __init__(self, spans):
        self.spans = spans
        self.starts, self.ends = zip(*spans)

    def __contains__(self, offset):
        s_idx = bisect(self.starts, offset)
        e_idx = bisect(self.ends, offset)
        return s_idx - e_idx == 1

    def covering(self, offset):
        s_idx = bisect(self.starts, offset)
        e_idx = bisect(self.ends, offset)
        if s_idx - e_idx == 1:
            return self.spans[e_idx]
