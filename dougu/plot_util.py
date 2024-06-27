import os
import matplotlib as mpl
if os.environ.get('DISPLAY') is None:  # NOQA
    mpl.use('Agg')  # NOQA

import matplotlib.pyplot as plt

from pathlib import Path

from .iters import is_non_string_iterable


class Figure:
    """Provides a context manager that automatically saves and closes
    a matplotlib plot.

    >>> with Figure("figure_name"):
    >>>     plt.plot(x, y)
    >>> # saves plot to {Figure.fig_dir}/{figure_name}.{Figure.file_type}

    When creating many figures with the same settings, e.g. plt.xlim(0, 100)
    and plt.ylim(0, 1.0), defaults can be set with:

    >>> Figure.set_defaults(xlim=(0, 100), ylim=(0, 1.0))
    >>> # context manager will call plt.xlim(0, 100) and plt.ylim(0, 1.0)
    """
    fig_dir = Path("out/fig")
    file_types = ["png", "pdf"]
    default_plt_calls = {}
    late_calls = ["xscale", "xlim", "yscale", "ylim"]  # order is important

    def __init__(
            self,
            name,
            figwidth=None,
            figheight=None,
            fontsize=12,
            invert_xaxis=False,
            invert_yaxis=False,
            out_dir=None,
            tight_layout=True,
            savefig_kwargs=None,
            **kwargs,
            ):
        if is_non_string_iterable(name):
            name = '.'.join(map(str, name))
        self.fig = plt.figure()
        if figwidth is not None:
            self.fig.set_figwidth(figwidth)
            phi = 1.6180
            self.fig.set_figheight(figheight or figwidth / phi)
        # params = {
        #     'figure.figsize': (figwidth, figheight or figwidth / phi),
        #     'axes.labelsize': fontsize,
        #     'axes.titlesize': fontsize,
        #     'legend.fontsize': fontsize,
        #     'xtick.labelsize': fontsize - 1,
        #     'ytick.labelsize': fontsize - 1,
        # }
        # mpl.rcParams.update(params)
        self.name = name
        self.plt_calls = {**kwargs}
        self.invert_xaxis = invert_xaxis
        self.invert_yaxis = invert_yaxis
        self.tight_layout = tight_layout
        self.savefig_kwargs = savefig_kwargs or {}
        self._out_dir = out_dir
        for attr, val in self.default_plt_calls.items():
            if attr not in self.plt_calls:
                self.plt_calls[attr] = val

    def __enter__(self):
        for attr, val in self.plt_calls.items():
            if attr in self.late_calls:
                continue
            try:
                getattr(plt, attr)(val)
            except Exception:
                getattr(plt, attr)(*val)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for attr in self.late_calls:
            if attr in self.plt_calls:
                getattr(plt, attr)(self.plt_calls[attr])
        if self.invert_xaxis:
            plt.gca().invert_xaxis()
        if self.invert_yaxis:
            plt.gca().invert_yaxis()
        if self.tight_layout:
            plt.tight_layout()
        for file_type in self.file_types:
            fname = f"{self.name}.{file_type}".replace(' ', '_')
            outfile = self.out_dir / fname
            print(outfile, self.savefig_kwargs)
            plt.savefig(outfile, **self.savefig_kwargs)
        plt.clf()

    @classmethod
    def set_defaults(cls, **kwargs):
        cls.default_plt_calls = kwargs
        for attr, val in kwargs.items():
            setattr(cls, attr, val)

    @classmethod
    def reset_defaults(cls):
        cls.default_plt_calls = {}

    @property
    def out_dir(self):
        return self._out_dir or self.fig_dir


def mpl_plot(
        plt_fn,
        *,
        name,
        filetype="png",
        **plt_kwargs):
    Figure.file_types = [filetype]
    Figure.set_defaults(**plt_kwargs)
    with Figure(name):
        return plt_fn()


linestyles = [
    "-", "--", "-.", ":",
    "-", "--", "-.", ":",
    "-", "--", "-.", ":",
    "-", "--", "-.", ":",
    "-", "--", "-.", ":",
    "-", "--", "-.", ":"]


try:
    from bokeh.palettes import Category20
    colors = Category20[20]
except ImportError:
    try:
        import seaborn as sns
        colors = sns.color_palette("muted")
    except ImportError:
        # https://gist.github.com/huyng/816622
        colors = [
            "348ABD", "7A68A6", "A60628",
            "467821", "CF4457", "188487", "E24A33",
            "348ABD", "7A68A6", "A60628",
            "467821", "CF4457", "188487", "E24A33",
            "348ABD", "7A68A6", "A60628",
            "467821", "CF4457", "188487", "E24A33",
            ]


# https://matplotlib.org/api/markers_api.html
markers = [
    ".",   # point
    ",",   # pixel
    "o",   # circle
    "v",   # triangle_down
    "^",   # triangle_up
    "<",   # triangle_left
    ">",   # triangle_right
    "1",   # tri_down
    "2",   # tri_up
    "3",   # tri_left
    "4",   # tri_right
    "8",   # octagon
    "s",   # square
    "p",   # pentagon
    "P",   # plus (filled)
    "*",   # star
    "h",   # hexagon1
    "H",   # hexagon2
    "+",   # plus
    "x",   # x
    "X",   # x (filled)
    "D",   # diamond
    "d",   # thin_diamond
    "|",   # vline
    "_",   # hline
    ]


# https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph  # NOQA
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    from mpl_toolkits import axes_grid1
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def get_cluster_colors(vectors, **clusterer_kwargs):
    """Clusters vectors and returns RGB colors for coloring each vector
    according to its cluster.
    Adapted from https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html#extract-the-clusters
    """
    import hdbscan
    import seaborn as sns
    clusterer = hdbscan.HDBSCAN(**clusterer_kwargs).fit(vectors)
    palette = sns.color_palette(n_colors=max(clusterer.labels_) + 1)
    cluster_colors = [
        sns.desaturate(palette[col], sat) if col >= 0 else (0.5, 0.5, 0.5)
        for col, sat in zip(clusterer.labels_, clusterer.probabilities_)
        ]
    # convert from [0, 1] floats to 8-bit RGB values
    return (np.array(cluster_colors) * 256).astype(int)
