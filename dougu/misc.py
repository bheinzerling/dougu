from pathlib import Path
from bisect import bisect
import re


def now_str():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def autocommit(
        repodir=".", glob_pattern="**/*.py", recursive=True, msg="autocommit",
        runid=None):
    """Commit all changes in repodir in files matching glob_pattern."""
    from git import Repo
    import glob
    repo = Repo(repodir)
    files = glob.glob(glob_pattern, recursive=recursive)
    try:
        repo.index.add(files)
    except OSError:
        import time
        import traceback
        traceback.print_exc()
        sleep_seconds = 7
        print(f"trying again in {sleep_seconds} seconds")
        time.sleep(sleep_seconds)
        repo.index.add(files)
    if repo.index.diff(repo.head.commit):
        repo.index.commit(msg + ("_" + str(runid) if runid else ""))
    sha = repo.head.object.hexsha
    return sha


class _Results():
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


def next_rundir(basedir=Path("out"), runid_fn="runid", log=None):
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


def args_to_str(args, positional_arg=None):
    """Convert an argparse.ArgumentParser object back into a string,
    e.g. for running an external command."""
    def val_to_str(v):
        if isinstance(v, list):
            return ' '.join(map(str, v))
        return str(v)

    def arg_to_str(k, v):
        k = f"--{k.replace('_', '-')}"
        if v is True:
            return k
        if v is False:
            return ""
        else:
            v = val_to_str(v)
        return k + " " + v

    if positional_arg:
        pos_args = val_to_str(args.__dict__[positional_arg]) + " "
    else:
        pos_args = ""

    return pos_args + " ".join([
        arg_to_str(k, v)
        for k, v in args.__dict__.items()
        if v is not None and k != positional_arg])


def str2bool(s):
    if s == "True":
        return True
    if s == "False":
        return False
    raise ValueError("Not a string representation of a boolean value:", s)


# source: https://stackoverflow.com/questions/16259923/how-can-i-escape-latex-special-characters-inside-django-templates  # NOQA
def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(
        re.escape(key)
        for key in sorted(conv.keys(), key=lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)


def df_to_latex(
        df,
        col_aligns="",
        header="",
        caption="",
        bold_max_cols=None,
        na_rep="-",
        supertabular=False,
        midrule_after=None):
    import numpy as np
    if bold_max_cols:
        max_cols_names = set(bold_max_cols)
    else:
        max_cols_names = set()
    if midrule_after is not None:
        if not hasattr(midrule_after, "__contains__"):
            midrule_after = set(midrule_after)

    def row_to_strings(row):
        if bold_max_cols is not None:
            max_val = row[bold_max_cols].max()
        else:
            max_val = np.NaN

        def fmt(key, val):
            if key in max_cols_names and val == max_val:
                return r"\textbf{" + str(val) + "}"
            elif not isinstance(val, str) and np.isnan(val):
                return na_rep
            else:
                return str(val)

        return [fmt(key, val) for key, val in row.items()]

    if not col_aligns:
        dtype2align = {
            np.dtype("int"): "r",
            np.dtype("float"): "r"}
        col_aligns = "".join([
            dtype2align.get(df[colname].dtype, "l")
            for colname in df.columns])
    if not header:
        header = " & ".join(map(tex_escape, df.columns)) + r"\\"
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        rows.append(" & ".join(row_to_strings(row)) + "\\\\\n")
        if midrule_after and i in midrule_after:
            rows.append("\midrule\n")
    if supertabular:

        latex_tbl = (r"""\tablehead{
\toprule
""" + header + "\n" + r"""\midrule
}
\tabletail{\bottomrule}
\bottomcaption{""" + caption + r"""}
\begin{supertabular}{""" + col_aligns + r"""}
""" + "".join(rows) + r"""\end{supertabular}
""")

    else:

        latex_tbl = (r"\begin{tabular}{" + col_aligns + r"""}
\toprule
""" + header + r"""
\midrule
""" + "".join(rows) + r"""\bottomrule
\end{tabular}
""")

    return latex_tbl


class SubclassRegistry:
    '''Mixin that automatically registers all subclasses of the
    given class.
    '''
    subclasses = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__.lower()] = cls
