from pathlib import Path
import os
import matplotlib as mpl
if os.environ.get('DISPLAY') is None:  # NOQA
    mpl.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import itertools
import numpy as np

# from pylab import rcParams
# rcParams['figure.figsize'] = (12, 12)
fontsize = 12
params = {
    'figure.figsize': (12, 12),
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'legend.fontsize': fontsize,
    'xtick.labelsize': fontsize - 1,
    'ytick.labelsize': fontsize - 1,
}
mpl.rcParams.update(params)


def histogram(
        values,
        *,
        name,
        filetype="png",
        plt_kwargs=None,
        hist_kwargs=None):
    n, bins, patches = mpl_plot(
        plt_fn=lambda: plt.hist(values, linewidth=10, **(hist_kwargs or {})),
        name=name,
        filetype=filetype,
        **(plt_kwargs or {}))
    return n, bins


def scatter(
        x,
        y,
        *,
        name,
        filetype="png",
        plt_kwargs=None,
        scatter_kwargs=None,
        fit_y=None,
        fit_plot_kwargs=None,
        legend=False):
    def plot_fit():
        if fit_y is not None:
            plt.plot(x, fit_y, **(fit_plot_kwargs or {}))

    def legend_fn():
        if legend:
            if isinstance(legend, dict):
                plt.legend(**legend)
            else:
                plt.legend()

    if isinstance(y, dict):
        def plt_fn():
            for label, _y in y.items():
                plt.scatter(x, _y, label=label, **(scatter_kwargs or {}))
            plot_fit()
            legend_fn()
    else:
        def plt_fn():
            plt.scatter(x, y, **(scatter_kwargs or {}))
            plot_fit()
            legend_fn()
    mpl_plot(
        plt_fn=plt_fn,
        name=name,
        filetype=filetype,
        **plt_kwargs)


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


# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-seq2seq-model  # NOQA
def plot_attention(
        input_labels, output_labels, attentions,
        out_colors=None, filepath=None):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_xticklabels([''] + output_labels, rotation=90)
    ax.set_yticklabels([''] + input_labels)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if out_colors:
        out_colors = ["k"] + out_colors
        assert len(out_colors) == 1 + len(output_labels), \
            f"got {len(out_colors)} colors for {len(output_labels)} labels"
        for xtick, color in zip(ax.get_xticklabels(), out_colors):
            xtick.set_color(color)

    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()
    plt.close()


# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html  # NOQA
def plot_confusion_matrix(
        cm, classes,
        normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,
        filepath=None):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(
            cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
                j, i, f"{cm[i, j]:.2f}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()
    plt.close()


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


def simple_imshow(
        matrix,
        cmap="viridis", figsize=(10, 10), aspect_equal=True, outfile=None,
        title=None, xlabel=None, ylabel=None,
        xtick_labels=None,
        ytick_labels=None,
        xtick_locs_labels=None,
        ytick_locs_labels=None,
        xtick_label_rotation='vertical',
        x_grid=None,
        y_grid=None,
        colorbar=True, scale="lin"):
    if aspect_equal and figsize[1] is None:
        matrix_aspect = matrix.shape[0] / matrix.shape[1]
        width = figsize[0]
        height = max(3, width * matrix_aspect)
        figsize = (width, height)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    if aspect_equal:
        ax.set_aspect('equal')
    if title:
        plt.title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    norm = matplotlib.colors.SymLogNorm(1) if scale == "log" else None
    im = plt.imshow(matrix, interpolation='nearest', cmap=cmap, norm=norm)
    if xtick_labels is not None:
        assert xtick_locs_labels is None
        locs = np.arange(0, len(xtick_labels))
        xtick_locs_labels = locs, xtick_labels
    if ytick_labels is not None:
        assert ytick_locs_labels is None
        locs = np.arange(0, len(ytick_labels))
        ytick_locs_labels = locs, ytick_labels
    if xtick_locs_labels is not None:
        plt.xticks(*xtick_locs_labels, rotation=xtick_label_rotation)
    if ytick_locs_labels is not None:
        plt.yticks(*ytick_locs_labels)
    if x_grid is not None or y_grid is not None:
        if x_grid is not None:
            ax.set_xticks(x_grid, minor=True)
        if y_grid is not None:
            ax.set_yticks(y_grid, minor=True)
        ax.grid(which="minor")
    if colorbar:
        add_colorbar(im)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
    plt.clf()


def embed_2d(emb, emb_method="UMAP", umap_n_neighbors=15, umap_min_dist=0.1):
    if emb_method == "UMAP":
        try:
            from umap import UMAP
        except ImportError:
            print("Please install umap to use emb_method='UMAP'")
            print("pip install umap-learn (NOT pip install umap)")
            print("https://github.com/lmcinnes/umap")
            raise
        proj = UMAP(
            init="random",
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist)
    else:
        import sklearn.manifold
        proj = getattr(sklearn.manifold, emb_method)()
    return proj.fit_transform(emb)


def plot_embeddings(
        emb, emb_method=None,
        labels=None, color=None, classes=None, class2color=None, title=None,
        outfile=None, cmap="viridis", max_labels=100,
        colorbar_ticks=None, reverse_colorbar=False, colorbar_label=None,
        label_fontpath=None,
        **scatter_kwargs):
    """
    Plot a scatterplot of the embeddings contained in emb.

    emb: an array with dim (n_embeddings x 2) or (n_embeddings x emb_dim).
    In the latter case an embedding method emb_method should be supplied
    to project from emb_dim to dim=2.

    emb_method: "UMAP", "TSNE", or any other algorithm in sklearn.manifold
    labels: Optional text labels for each embedding
    color: Optional color for each embedding, according to which it will be
    colored in the plot.
    classes:  Optional class for each embedding, according to which it will
    be colored in the plot.
    class2color: A map which determines the color assigned to each class
    outfile: If provided, save plot to this file instead of showing it
    cmap: colormap
    max_labels: maximum number of labels to be displayed
    """
    from matplotlib.ticker import NullFormatter
    if emb_method:
        x, y = embed_2d(emb, emb_method).T
    else:
        x, y = emb.T
    figsize = (14, 12) if color is not None else (12, 12)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    if not scatter_kwargs:
        scatter_kwargs = dict(marker="o", s=1, alpha=1)
    if classes is not None:
        for cls in set(classes):
            i = (classes == cls).nonzero()
            ax.scatter(x[i], y[i], label=cls, **scatter_kwargs)
    elif color is not None:
        sc = ax.scatter(x, y, c=color, cmap=cmap, **scatter_kwargs)
        cb = fig.colorbar(sc, ticks=colorbar_ticks)
        if reverse_colorbar:
            cb.ax.invert_yaxis()
        if colorbar_label:
            cb.set_label(colorbar_label)
    else:
        ax.scatter(x, y, **scatter_kwargs)

    if labels is not None:
        if label_fontpath:
            import matplotlib.font_manager as fm
            fontproperties = fm.FontProperties(fname=label_fontpath)
        else:
            fontproperties = None
        n_labels = len(labels)
        for i in range(len(emb)):
            if (
                    max_labels < 0 or
                    n_labels <= max_labels or
                    not i % (n_labels // max_labels)):
                ax.annotate(
                    labels[i], (x[i], y[i]), alpha=0.76, size=10,
                    fontproperties=fontproperties)
    if title:
        plt.title(title)
    plt.axis('tight')
    if classes is not None:
        plt.legend(loc='best', scatterpoints=1, markerscale=5, fontsize=10)
    plt.tight_layout()
    if outfile:
        plt.savefig(str(outfile))
    else:
        plt.show()


def plot_dendrogram(dist, labels, outfile=None, method="centroid"):
    from scipy.cluster import hierarchy
    fig = plt.figure(figsize=(50, 45))
    # dendrogram
    axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    Y = hierarchy.linkage(dist, method=method)
    Z = hierarchy.dendrogram(
        Y, orientation='right', labels=labels, leaf_font_size=10)
    # distance matrix
    index = Z['leaves']
    D = dist[index, :]
    D = D[:, index]
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.8])
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    # colorbar
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    plt.colorbar(im, cax=axcolor)

    if outfile:
        fig.savefig(str(outfile))
    else:
        fig.show()
    plt.close(fig)


def plot_embeddings_bokeh(
        emb,
        emb_method=None,
        classes=None, labels=None, color=None,
        color_category=None,
        cmap=None, cmap_reverse=False,
        colorbar=False, colorbar_ticks=None,
        outfile=None, title=None,
        scatter_labels=False,
        **circle_kwargs):
    """
    Creates an interactive scatterplot of the embeddings contained in emb,
    using the bokeh library.

    emb: an array with dim (n_embeddings x 2) or (n_embeddings x emb_dim).
    In the latter case an embedding method emb_method should be supplied
    to project from emb_dim to dim=2.

    emb_method: "UMAP", "TSNE", or any other algorithm in sklearn.manifold
    labels: Optional text labels for each embedding
    color: Optional color for each embedding, according to which it will be
    colored in the plot.
    classes:  Optional class for each embedding, according to which it will
    be colored in the plot.
    outfile: If provided, save plot to this file instead of showing it
    cmap: colormap
    title: optional title of the plot
    """
    from bokeh.plotting import figure, output_file, show, save
    from bokeh.models import (
        ColumnDataSource, CategoricalColorMapper, LinearColorMapper,
        ColorBar, FixedTicker, Text)
    from bokeh.palettes import Category20, Viridis256, viridis

    if emb_method:
        emb = embed_2d(emb, emb_method)

    if outfile:
        output_file(outfile)

    source_dict = dict(x=emb[:, 0], y=emb[:, 1])
    if classes is not None:
        source_dict["cls"] = classes
    if labels is not None:
        source_dict["label"] = labels
    if color is not None:
        source_dict["color"] = color
    source = ColumnDataSource(source_dict)
    if classes is not None and color is None:
        n_classes = len(set(classes))
        if n_classes <= 20:
            if n_classes <= 2:
                palette = Category20[3]
                palette = [palette[0], palette[-1]]
            else:
                palette = Category20[n_classes]
        else:
            palette = viridis(n_classes)
        color_conf = {
            "field": "cls",
            "transform": CategoricalColorMapper(
                factors=list(set(classes)),
                palette=palette)}
    elif color is not None:
        if cmap is not None:
            if isinstance(cmap, str):
                import bokeh.palettes
                # matplotib suffix for reverse color maps
                if cmap.endswith("_r"):
                    cmap_reverse = True
                    cmap = cmap[:-2]
                cmap = getattr(bokeh.palettes, cmap)
            elif isinstance(cmap, dict):
                cmap = cmap[max(cmap.keys())]
        else:
            cmap = Viridis256
        if cmap_reverse:
            cmap.reverse()
        color_mapper = LinearColorMapper(cmap)
        color_conf = {
            "field": "color",
            "transform": color_mapper}
        if colorbar:
            if colorbar_ticks:
                ticker = FixedTicker(ticks=colorbar_ticks)
            else:
                ticker = None
            colorbar = ColorBar(
                color_mapper=color_mapper, ticker=ticker)
    else:
        color_conf = "red"
    tools = "crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"
    p = figure(tools=tools, sizing_mode='scale_both')
    if title:
        p.title.text = title
    if labels is not None and scatter_labels:
        glyph = Text(
            x="x", y="y", text="label", angle=0.0,
            text_color=color_conf, text_alpha=0.95, text_font_size="8pt")
        p.add_glyph(source, glyph)
    else:
        p.circle(
            x='x', y='y',
            source=source,
            color=color_conf,
            legend='cls' if classes is not None else None,
            **circle_kwargs)
    if labels is not None:
        from bokeh.models import HoverTool
        from collections import OrderedDict
        hover = p.select(dict(type=HoverTool))
        hover_entries = [
            ("label", "@label"),
            ("(x, y)", "(@x, @y)"),
            ]
        if color is not None and color_category:
            hover_entries.append((color_category, "@color"))
        hover.tooltips = OrderedDict(hover_entries)
    if colorbar:
        assert color is not None
        p.add_layout(colorbar, 'right')
    if outfile:
        save(p)
    else:
        show(p)


class Figure():
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
    fig_dir = Path("out/figs")
    file_types = ["png", "pdf"]
    default_plt_calls = {}
    late_calls = ["xscale", "xlim", "yscale", "ylim"]  # order is important

    def __init__(
            self, name,
            figwidth=6, figheight=None, fontsize=12,
            invert_xaxis=False, invert_yaxis=False,
            **kwargs):
        self.fig = plt.figure()
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
        for attr, val in self.default_plt_calls.items():
            if attr not in self.plt_calls:
                self.plt_calls[attr] = val

    def __enter__(self):
        for attr, val in self.plt_calls.items():
            # if attr in self.late_calls:
            #     continue
            try:
                getattr(plt, attr)(val)
            except:
                getattr(plt, attr)(*val)

        return self.fig

    def __exit__(self, exc_type, exc_val, exc_tb):
        # for attr in self.late_calls:
        #     if attr in self.plt_calls:
        #         print(attr, self.plt_calls[attr])
        #         getattr(plt, attr)(self.plt_calls[attr])
        if self.invert_xaxis:
            plt.gca().invert_xaxis()
        if self.invert_yaxis:
            plt.gca().invert_yaxis()
        plt.tight_layout()
        for file_type in self.file_types:
            outfile = self.fig_dir / f"{self.name}.{file_type}"
            plt.savefig(outfile)
        plt.clf()

    @classmethod
    def set_defaults(cls, **kwargs):
        cls.default_plt_calls = kwargs
        for attr, val in kwargs.items():
            setattr(cls, attr, val)

    @classmethod
    def reset_defaults(cls):
        cls.default_plt_calls = {}


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


if __name__ == "__main__":
    plot_attention(
        "1 2 3 4".split(),
        "a b c d".split(),
        np.random.rand(4, 4),
        out_colors="r g b r".split())
