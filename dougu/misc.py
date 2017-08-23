from datetime import datetime
import glob
from pathlib import Path

import joblib
from pandas import DataFrame as df
from git import Repo


def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def autocommit(
        repodir=".", glob_pattern="**/*.py", recursive=True, msg="autocommit"):
    """Commit all changes in repodir in files matching glob_pattern."""
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
        self.fname_pkl = Path(str(file) + ".df.pkl")
        self.fname_txt = Path(str(file) + ".txt")
        self.fname_stats = Path(str(file) + ".overall_stats")
        if self.fname_pkl.exists():
            self.results = joblib.load(self.fname_pkl)
        else:
            self.results = df()

    def append(self, results_row):
        row = df([results_row])
        if self.results is None:
            self.results = row
        else:
            self.results = self.results.append(row, ignore_index=True)
        joblib.dump(self.results, self.fname_pkl)
        with self.fname_txt.open("a", encoding="utf8") as out:
            out.write("\t".join(list(map(str, results_row.values()))) + "\n")
        with self.fname_stats.open("w", encoding="utf8") as out:
            out.write(str(self.results.describe()))

    def max(self, score_fun=lambda df: df.mean(1)):
        return float(score_fun(self.results).max())
