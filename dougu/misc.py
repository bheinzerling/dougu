from pathlib import Path
from bisect import bisect
import re
from collections import defaultdict


from .io import mkdir


def now_str():
    """String representation of the current datetime."""
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
    """Wrapper around a pandas DataFrame to keep track of
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
    """Get the next run id by incrementing the id stored in a file.
    (faster than taking the maximum over all subdirs)"""
    try:
        with file.open() as f:
            runid = int(f.read()) + 1
    except FileNotFoundError:
        runid = 0
    with file.open("w") as out:
        out.write(str(runid))
    return runid


def next_rundir(basedir=Path("out"), runid_fname="runid", log=None):
    """Create a directory for running an experiment."""
    runid = get_and_increment_runid(basedir / runid_fname)
    rundir = basedir / str(runid)
    rundir.mkdir(exist_ok=True, parents=True)
    if log:
        log.info(f"rundir: {rundir.resolve()}")
    return rundir


def color_range(start_color, end_color, steps=10, cformat=lambda c: c.hex_l):
    """Thin wrapper around the range_to method in the colour library."""
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

    def __iter__(self):
        return iter(self.spans)

    def __contains__(self, offset):
        s_idx = bisect(self.starts, offset)
        e_idx = bisect(self.ends, offset)
        return s_idx - e_idx == 1

    def covering(self, offset):
        s_idx = bisect(self.starts, offset)
        e_idx = bisect(self.ends, offset)
        if s_idx - e_idx == 1:
            return self.spans[e_idx], e_idx


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
    """Converts a pandas dataframe into a latex table.

        :col_aligns: Optional format string for column alignments in the
                     tabular environment, e.g. 'l|c|rrr'. Will default
                     to right-align for numeric columns and left-align
                     for everthing else.
        :header: Optional header row. Default is the df column names.
        :bold_max_cols: Indexes of the columns among which the maximum
                        value will be bolded.
        :na_rep: string representation of missing values.
        :supertabular: Whether to use the supertabular package (for long
                       tables).
        :midrule_after: Indexes of the rows after which a midrule should
                        be inserted.
    """
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
            rows.append(r"\midrule\n")
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
    classes = dict()
    subclasses = defaultdict(set)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.classes[cls.__name__.lower()] = cls
        for super_cls in cls.__mro__:
            if super_cls == cls:
                continue
            SubclassRegistry.subclasses[super_cls].add(cls)

    @staticmethod
    def get(cls_name):
        return SubclassRegistry.classes[cls_name]

    @classmethod
    def get_subclasses(cls):
        return SubclassRegistry.subclasses.get(cls, {})


def auto_debug():
    """Automatically start a debugger when an exception occcurs."""
    import sys

    def info(ty, value, tb):
        if (
                ty is KeyboardInterrupt or
                hasattr(sys, 'ps1') or
                not sys.stderr.isatty()):
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(ty, value, tb)
        else:
            from IPython.core import ultratb
            ultratb.FormattedTB(
                mode='Verbose', color_scheme='Linux', call_pdb=1)()

    sys.excepthook = info


def add_jobid(args):
    """Add jobid to argparser object if the current program is a
    SGE/UGE batch job."""
    import os
    is_batchjob = (
        'JOB_SCRIPT' in os.environ and os.environ['JOB_SCRIPT'] != 'QRLOGIN')
    if is_batchjob and 'JOB_ID' in os.environ:
        args.jobid = os.environ['JOB_ID']


def make_and_set_rundir(args):
    """Make rundir and set args.rundir to the corresponding path."""
    if args.runid is not None:
        args.rundir = mkdir(args.outdir / args.runid)
    else:
        args.rundir = next_rundir()
        args.runid = args.rundir.name


def conf_hash(conf, fields=None):
    """Return a hash value for the a configuration object, e.g. an
    argparser instance. Useful for creating unique filenames based on
    the given configuration."""
    if fields is None:
        d = conf.__dict__
    else:
        d = {k: getattr(conf, k) for k in fields}
    import hashlib
    return hashlib.md5(bytes(repr(d), encoding='utf8')).hexdigest()
