import os
import matplotlib as mpl
if os.environ.get('DISPLAY') is None:  # NOQA
    mpl.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import itertools
import numpy as np
from seaborn.utils import relative_luminance

from .embeddingutil import embed_2d
from .plot_bokeh import plot_embeddings_bokeh  # NOQA
from .plot_util import (
    mpl_plot,
    add_colorbar,
    Figure,
    )

# from pylab import rcParams
# rcParams['figure.figsize'] = (12, 12)
# fontsize = 12
# params = {
#     'figure.figsize': (12, 12),
#     'axes.labelsize': fontsize,
#     'axes.titlesize': fontsize,
#     'legend.fontsize': fontsize,
#     'xtick.labelsize': fontsize - 1,
#     'ytick.labelsize': fontsize - 1,
# }
# mpl.rcParams.update(params)


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


# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-seq2seq-model  # NOQA
def plot_attention(
        input_labels, output_labels, attentions,
        out_colors=None, filepath=None):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.matshow(attentions, cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    # Set up axes
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_xticklabels([''] + output_labels, rotation=90)
    ax.set_yticklabels([''] + input_labels, rotation=45)
    ax.xaxis.set_ticks_position('bottom')

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if out_colors:
        out_colors = ["k"] + out_colors
        assert len(out_colors) == 1 + len(output_labels), \
            f"got {len(out_colors)} colors for {len(output_labels)} labels"
        for xtick, color in zip(ax.get_xticklabels(), out_colors):
            xtick.set_color(color)

    plt.tight_layout()

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


def simple_imshow(
        matrix,
        cmap="viridis",
        figsize=(10, 10),
        aspect_equal=True,
        outfile=None,
        title=None,
        xlabel=None,
        ylabel=None,
        xticks=True,
        yticks=True,
        xtick_labels=None,
        ytick_labels=None,
        xtick_locs_labels=None,
        ytick_locs_labels=None,
        tick_labelsize=None,
        xtick_label_rotation='vertical',
        xgrid=None,
        ygrid=None,
        colorbar=True,
        scale="lin",
        colorbar_range=None,
        cbar_title=None,
        bad_color='white',
        origin='upper',
        cell_text=None,
        cell_text_color=None,
        ):
    if aspect_equal and figsize is not None and figsize[1] is None:
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
    cmap = mpl.cm.get_cmap(cmap)
    if bad_color is not None:
        cmap.set_bad(bad_color)
    im = plt.imshow(
        matrix, interpolation='nearest', cmap=cmap, norm=norm, origin=origin)
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
    if xgrid is not None or ygrid is not None:
        if xgrid is not None:
            ax.set_xticks(xgrid, minor=True)
        if ygrid is not None:
            ax.set_yticks(ygrid, minor=True)
        ax.grid(which="minor")
    if xticks is not True:
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,)         # ticks along the top edge are off
        ax.set_xticks([])
    if yticks is not True:
        plt.tick_params(
            axis='y',          # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,)         # ticks along the top edge are off
        ax.set_yticks([])
    if colorbar:
        cbar = add_colorbar(im)
        if colorbar_range is not None:
            plt.clim(*colorbar_range)
        if cbar_title:
            cbar.ax.set_ylabel(cbar_title, labelpad=3, rotation=90)
    if tick_labelsize is not None:
        ax.xaxis.set_tick_params(labelsize=tick_labelsize)
        ax.yaxis.set_tick_params(labelsize=tick_labelsize)
    if cell_text is not None:
        for (i, j), val in np.ndenumerate(matrix):
            color = cmap(val)
            # source: https://github.com/mwaskom/seaborn/blob/6890b315d00b74f372bc91f3929c803837b2ddf1/seaborn/matrix.py#L258
            lum = relative_luminance(color)
            if cell_text_color is None:
                _cell_text_color = ".15" if lum > .408 else "w"
            else:
                _cell_text_color = cell_text_color
            ax.text(j, i, cell_text[i, j], color=_cell_text_color, ha='center', va='center')
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)


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


def plot_dendrogram(
        dist,
        labels,
        outfile=None,
        method="centroid",
        figsize=(50, 45),
        font_size=10,
        cmap='magma_r',
        ):
    from scipy.cluster import hierarchy
    fig = plt.figure(figsize=figsize)
    # dendrogram
    axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    Y = hierarchy.linkage(dist, method=method)
    Z = hierarchy.dendrogram(
        Y, orientation='right', labels=labels, leaf_font_size=font_size)
    # distance matrix
    index = Z['leaves']
    D = dist[index, :]
    D = D[:, index]
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.8])
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=cmap)
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


def plot_graph(graph=None, edges=None, *, outfile=None, name=''):
    """Create a plot of `graph` or the graph specified by `edges` and save it.

    graph: a networkx graph
    edges:
        list of (source, target) tuples or list of
        (source, edge_label, target) triples
    outfile: file to which the plot will be saved in HTML format
    """
    assert graph is not None or edges is not None
    if graph is None:
        from .graph import graph_from_edges
        graph = graph_from_edges(edges)
    from pyvis import network as net
    n = net.Network(height='100%', width='70%', directed=True)
    n.from_nx(graph)
    n.show_buttons(filter_=['physics'])
    if outfile:
        n.write_html(str(outfile))
    else:
        n.show(name)


def plot(
        *,
        data=None,
        x=None,
        y=None,
        group_col=None,
        title=None,
        xaxis_title=None,
        yaxis_title=None,
        group_title=None,
        color_col=None,
        color_discrete=False,
        legend_loc='upper right',
        palette="viridis",
        kind='line',
        norm=None,
        vmin=None,
        vcenter=None,
        vmax=None,
        with_marginals=False,
        xlim=None,
        ylim=None,
        fig_kwargs=None,
        **plot_kwargs,
        ):
    fig_kwargs = fig_kwargs or dict()
    import seaborn as sns
    plot_fn = getattr(sns, f'{kind}plot')
    with Figure(title, **fig_kwargs):
        long_xtick_labels = False
        if data is None and x is None:
            x = list(range(len(y)))
        if x is not None:
            if data is not None:
                max_len = data[x].astype(str).str.len().max()
            else:
                max_len = max(map(len, map(str, x)))
            long_xtick_labels = max_len > 5
            if long_xtick_labels:
                plt.figure(figsize=(12, 12))

        hue_col = group_col or color_col
        if hue_col is not None:
            hue_order = data[hue_col].sort_values().unique().tolist()
            hue_col_is_numeric = data[hue_col].dtype != 'O'
        else:
            hue_col_is_numeric = False

        if norm is None and hue_col_is_numeric:
            if vcenter is not None:
                from matplotlib.colors import TwoSlopeNorm
                norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            else:
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                # data[colorbar_col].min(), data[colorbar_col].max())

        if kind in {'scatter', 'joint'} or hue_col_is_numeric:
            style = None
        else:
            style = hue_col
        if with_marginals:
            g = sns.JointGrid(data=data, x=x, y=y)
            g.plot_joint(
                plot_fn,
                data=data,
                hue=hue_col,
                hue_norm=norm,
                style=style,
                palette=palette,
                **plot_kwargs,
                )
            g.plot_marginals(sns.histplot)
            ax = g.ax_joint
            cax = g.fig.add_axes([.74, .13, .015, .12])
        else:
            ax = plot_fn(
                data=data,
                x=x,
                y=y,
                hue=hue_col,
                hue_norm=norm,
                style=style,
                palette=palette,
                **plot_kwargs,
                )
            cax = ax

        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)

        if plot_kwargs.get('legend', True) and color_discrete and hue_col is not None:
            # sort labels in legend
            handles, labels = plt.gca().get_legend_handles_labels()
            label2handle = dict(zip(labels, handles))
            new_labels = list(map(str, hue_order))
            new_handles = [label2handle[label] for label in new_labels]
            cmap = sns.color_palette(palette, as_cmap=True)
            for val, old_handle, new_handle in zip(hue_order, handles, new_handles):
                color = old_handle._color
                new_handle.set_color(color)
            plt.legend(
                new_handles,
                new_labels,
                loc=legend_loc,
                title=hue_col,
                )

        if long_xtick_labels:
            plt.xticks(rotation=90)
        if color_col and not color_discrete:
            if color_col == group_col or group_col is None:
                legend = ax.get_legend()
                if legend:
                    legend.remove()
            cmap = sns.color_palette(palette, as_cmap=True)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = plt.colorbar(sm)
            cbar.set_label(color_col, fontsize='small')


simple_plot = plot


def plot_distribution(
        data=None,
        *,
        x=None,
        title=None,
        ):
    import seaborn as sns
    with Figure(name=title):
        sns.violinplot(
            data=data,
            x=x,
            cut=0,
            )
