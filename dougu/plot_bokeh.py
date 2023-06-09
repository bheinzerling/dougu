from .decorators import cached_property


def get_palette(categories, cmap=None):
    from bokeh.palettes import (
        Category20,
        Category20b,
        Category20c,
        viridis,
        )
    n_cat = len(set(categories))
    if cmap is not None:
        return cmap[n_cat]
    if n_cat <= 20:
        if n_cat <= 2:
            palette = Category20[3]
            return [palette[0], palette[-1]]
        else:
            return Category20[n_cat]
    if n_cat <= 40:
        return Category20[20] + Category20b[20]
    if n_cat <= 60:
        return Category20[20] + Category20b[20] + Category20c[20]
    return viridis(n_cat)


class BokehFigure:
    """
    data: Pandas dataframe or column dictionary
    plot_width: width of the bokeh plot
    plot_height: height of the bokeh plot
    reuse_figure: add plot to this figure instead of creating a new one
    sizing_mode: layout parameter
        see: https://docs.bokeh.org/en/latest/docs/reference/layouts.html#layout
    width: the width of the figure in pixels
    height: the height of the figure in pixels
    figure_kwargs: additional arguments that will be passed to
    bokeh.plotting.figure
    title: optional title of the plot
    colorbar: specifies whether to add a colorbar or not
    colorbar_title: optional colorbar title text
    colorticks: supply a list of tick values or set to False to disable
    labels: Optional text labels for each data point (or the name of a column
    if `data` is supplied)
    color: Optional color index for each datapoint, according to which it
    will be assigned a color (or the name of a column if `data` is supplied)
    raw_color: set this to true if `color` contains raw (RGB) color values
    instead of color indexes
    color_categorical: set to True to force a categorical color map (use in
    case the automatic colormap selection fails to select a categorical color
    map)
    cmap_reverse: set to True to reverse the chosen colormap
    classes:  Optional class for each datapoint, according to which it will
    be colored in the plot (or the name of a column if `data` is supplied).
    cmap: name of a bokeh/colorcet colormap or a dictionary from values to
    color names
    tooltip_fields: column names for `data` or a column dictionary with data
    for each data point that will be used to add tooltip information
    plot_kwargs: additional arguments that will be pass to the bokeh glyph
    """
    def __init__(
            self,
            *,
            data=None,
            plot_width=None,
            plot_height=None,
            reuse_figure=None,
            sizing_mode='stretch_both',
            width=800,
            height=800,
            figure_kwargs=None,
            title=None,
            xaxis_label=None,
            yaxis_label=None,
            colorbar=False,
            colorbar_title=None,
            colorbar_ticks=True,
            labels=None,
            color=None,
            color_category=None,
            raw_colors=None,
            color_categorical=False,
            cmap=None,
            cmap_reverse=False,
            classes=None,
            class_category=None,
            tooltip_fields=None,
            **plot_kwargs,
            ):

        self.data = data

        def maybe_data_column(key_or_values):
            if isinstance(key_or_values, str) and self.data is not None:
                key = key_or_values
                values = self.data[key]
            else:
                values = key_or_values
            return values

        self.classes = classes
        self.class_category = class_category
        self.color = maybe_data_column(color)
        self.color_category = color_category
        self.raw_colors = raw_colors
        self.color_categorical = color_categorical
        self._cmap = cmap
        self.cmap_reverse = cmap_reverse
        self.plot_kwargs = plot_kwargs
        self.tooltip_fields = tooltip_fields

        self.labels = maybe_data_column(labels)
        self.make_figure(
            plot_width=plot_width,
            plot_height=plot_height,
            reuse_figure=reuse_figure,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            **(figure_kwargs or {}),
            )
        self.add_title(title)
        self.add_axis_labels(xaxis_label=xaxis_label, yaxis_label=yaxis_label)
        self.add_tooltips()
        self.add_colorbar(
            colorbar=colorbar,
            colorbar_ticks=colorbar_ticks,
            title=colorbar_title,
            )

    def set_labels(self, labels):
        if isinstance(labels, str) and self.data is not None:
            self.labels = self.data[labels]
        else:
            self.labels = labels

    def make_figure(
            self,
            plot_width=None,
            plot_height=None,
            reuse_figure=None,
            sizing_mode='stretch_both',
            width=800,
            height=800,
            **figure_kwargs,
            ):
        from bokeh.plotting import figure
        if plot_width is not None:
            figure_kwargs['plot_width'] = plot_width
        if plot_height is not None:
            figure_kwargs['plot_height'] = plot_height
        if reuse_figure is None:
            self.figure = figure(
                tools=self.tools_str,
                sizing_mode=sizing_mode,
                width=width,
                height=height,
                **figure_kwargs,
                )
        else:
            self.figure = reuse_figure

    def add_title(self, title):
        if title:
            self.figure.title.text = title

    def add_axis_labels(self, xaxis_label=None, yaxis_label=None):
        self.figure.xaxis.axis_label = xaxis_label
        self.figure.yaxis.axis_label = yaxis_label

    @property
    def hover_tool_names(self):
        return None

    def add_tooltips(self):
        from bokeh.models import HoverTool
        hover = self.figure.select(dict(type=HoverTool))
        hover_entries = []
        if self.labels is not None:
            hover_entries.append(("label", "@label{safe}"))
        hover_entries.append(("(x, y)", "(@x, @y)"))
        if self.color is not None and self.color_category:
            hover_entries.append((self.color_category, "@color"))
        if self.classes is not None and self.class_category:
            hover_entries.append((self.class_category, "@class"))
        if self.tooltip_fields:
            for field in self.tooltip_fields:
                hover_entries.append((field, "@" + field))
        hover.tooltips = dict(hover_entries)

    def add_colorbar(
            self,
            colorbar=False,
            colorbar_ticks=None,
            title=None
            ):
        if colorbar:
            from bokeh.models import ColorBar, BasicTicker, FixedTicker
            if colorbar_ticks:
                if colorbar_ticks is True:
                    ticker = BasicTicker()
                else:
                    ticker = FixedTicker(ticks=colorbar_ticks)
                ticker_dict = dict(ticker=ticker)
            else:
                ticker_dict = {}
            colorbar = ColorBar(
                color_mapper=self.color_mapper,
                title=title,
                height=240,
                **ticker_dict,
                )
            self.figure.add_layout(colorbar, 'right')

    @cached_property
    def tools(self):
        return [
            'crosshair',
            'pan',
            'wheel_zoom',
            'box_zoom',
            'reset',
            'hover',
            ]

    @cached_property
    def tools_str(self):
        return ','.join(self.tools)

    def save_or_show(self, outfile=None):
        if outfile:
            self.save(outfile=outfile)
        else:
            self.show()

    def save(self, outfile=None, write_png=False):
        from bokeh.plotting import output_file, save
        output_file(outfile)
        save(self.figure)
        if write_png:
            from bokeh.io import export_png
            png_file = outfile.with_suffix('.png')
            export_png(self.figure, filename=png_file)

    def show(self):
        from bokeh.plotting import show
        show(self.figure)

    @cached_property
    def source(self):
        from bokeh.models import ColumnDataSource
        source_dict = self.source_dict
        if self.labels is not None:
            source_dict["label"] = self.labels

        if self.raw_colors is not None:
            assert self.color is None
            if any(isinstance(c, str) for c in self.raw_colors):
                assert all(isinstance(c, str) for c in self.raw_colors)
            else:
                assert all(len(c) == 3 for c in self.raw_colors)
                assert self.cmap is None
                from bokeh.colors import RGB
                raw_colors = [RGB(*c) for c in self.raw_colors]
            source_dict["color"] = raw_colors
            self.color_conf = {"field": "color"}
        elif self.color is not None:
            self.color_conf = {
                "field": "color",
                "transform": self.color_mapper}
            source_dict["color"] = self.color
        else:
            self.color_conf = "red"

        if self.classes is not None:
            source_dict["class"] = self.classes
        if self.tooltip_fields:
            if hasattr(self.tooltip_fields, 'items'):
                for k, v in self.tooltip_fields.items():
                    source_dict[k] = v
            elif self.data is not None:
                for f in self.tooltip_fields:
                    source_dict[f] = self.data[f]
        source = ColumnDataSource(source_dict)
        return source

    @cached_property
    def source_dict(self):
        raise NotImplementedError()

    def plot(self):
        self._plot()
        return self

    def _plot(self):
        raise NotImplementedError()

    @cached_property
    def color_mapper(self):
        from bokeh.models import (
            CategoricalColorMapper,
            LinearColorMapper,
            )
        if self.color is not None and any(isinstance(c, str) for c in self.color):
            assert all(isinstance(c, str) for c in self.color)
            palette = get_palette(self.color, cmap=self.cmap)
            return CategoricalColorMapper(
                    factors=sorted(set(self.color)),
                    palette=palette)
        else:
            cmap = self.cmap
            if cmap is None:
                cmap = self.cmap_default
            elif isinstance(self.cmap, dict):
                cmap = self.cmap[max(self.cmap.keys())]
            return LinearColorMapper(cmap)

    @cached_property
    def cmap_default(self):
        from bokeh.palettes import Viridis256
        return Viridis256

    @cached_property
    def cmap(self):
        cmap = self._cmap
        cmap_reverse = self.cmap_reverse
        if cmap is not None:
            if isinstance(cmap, str):
                import bokeh.palettes
                # matplotib suffix for reverse color maps
                if cmap.endswith("_r"):
                    cmap_reverse = True
                    cmap = cmap[:-2]
                palette = getattr(bokeh.palettes, cmap, None)
                if palette is None:
                    import colorcet
                    palette = colorcet.palette[cmap]
                assert palette
                cmap = palette
            elif isinstance(cmap, dict):
                cmap = cmap[max(cmap.keys())]
            if cmap_reverse:
                if isinstance(cmap, dict):
                    new_cmap = {}
                    for k, v in cmap.items():
                        v = list(v)
                        v.reverse()
                        new_cmap[k] = v
                    cmap = new_cmap
                else:
                    cmap = list(cmap)
                    cmap.reverse()
        return cmap


class ScatterBokeh(BokehFigure):
    """Creates an interactive bokeh scatter plot.

    x: sequence of values or a column name if `data` is given
    y: sequence of values or a column name if `data` is given
    scatter_labels: set to True to scatter labels, i.e.,
    draw text instead of points
    data: Pandas dataframe or column dictionary
    """

    def __init__(self, *, x, y, scatter_labels=False, data=None, **kwargs):
        super().__init__(data=data, **kwargs)
        if data is not None:
            x = data[x]
            y = data[y]
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.scatter_labels = scatter_labels

    @cached_property
    def source_dict(self):
        return dict(x=self.x, y=self.y)

    @property
    def glyph_name(self):
        return 'scatter'

    @property
    def hover_tool_names(self):
        return [self.glyph_name]

    def _plot(self):
        if self.scatter_labels:
            assert self.labels is not None
            from bokeh.models import Text
            glyph = Text(
                x="x",
                y="y",
                text="label",
                angle=0.0,
                text_color=self.color_conf,
                text_alpha=0.95,
                text_font_size="8pt",
                name=self.glyph_name,
                **self.plot_kwargs,
                )
            self.figure.add_glyph(self.source, glyph)
        else:
            plot_kwargs = dict(
                x='x',
                y='y',
                source=self.source,
                color=self.color_conf,
                name=self.glyph_name,
                **self.plot_kwargs,
                )
            if self.classes is not None:
                legend_field = 'class'
            else:
                legend_field = None
            if legend_field:
                plot_kwargs['legend_field'] = legend_field
                # sort by color field to order the legend entries nicely
                sorted_source = self.source.to_df().sort_values(legend_field)
                plot_kwargs['source'] = self.source.from_df(sorted_source)

            self.figure.circle(**plot_kwargs)


class CliqueScatterBokeh(ScatterBokeh):
    """Creates an interactive bokeh scatter plot, where each point
    is assumed to be part of a clique. All points in a clique will
    be connected with lines.

    clique_ids: column name (if `data` is supplied) or sequence of
    values according to which each point will be assigned to a clique.
    data: Pandas dataframe or column dictionary
    """
    def __init__(self, *, clique_ids, data=None, **kwargs):
        super().__init__(data=data, **kwargs)
        if data is not None:
            clique_ids = data[clique_ids]
        self.clique_ids = clique_ids

    @cached_property
    def source_dict(self):
        return super().source_dict | dict(clique_ids=self.clique_ids)

    @cached_property
    def edges(self):
        from .iters import groupby, unordered_pairs
        clique_id2points = groupby(self.clique_ids, zip(self.x, self.y))
        assert sum(map(len, clique_id2points.values())) == len(self.x)
        return [
            {
                'clique_id': clique_id,
                'x0': p0[0],
                'y0': p0[1],
                'x1': p1[0],
                'y1': p1[1],
            }
            for clique_id, points in clique_id2points.items()
            for p0, p1 in unordered_pairs(points)
            ]

    @cached_property
    def edges_source(self):
        from bokeh.models import ColumnDataSource
        from .iters import concat_dict_values
        return ColumnDataSource(concat_dict_values(self.edges))

    @cached_property
    def edge_color(self):
        return '#f4a582'

    @cached_property
    def edge_width(self):
        return 0.5

    def _plot(self):
        self._plot_edges()
        super()._plot()

    def _plot_edges(self):
        from bokeh.models import Segment
        glyph = Segment(
            x0='x0',
            y0='y0',
            x1='x1',
            y1='y1',
            line_color=self.edge_color,
            line_width=self.edge_width,
            )
        self.figure.add_glyph(self.edges_source, glyph)


def plot_embeddings_bokeh(
        emb,
        emb_method='UMAP',
        outfile=None,
        **kwargs,
        ):
    """
    Creates an interactive scatterplot of the embeddings contained in emb,
    using the bokeh library.
    emb: an array with dim (n_embeddings x 2) or (n_embeddings x emb_dim).
    In the latter case emb_method will be applied to project from emb_dim
    to dim=2.
    emb_method: "UMAP", "TSNE", or any other algorithm in sklearn.manifold
    outfile: If provided, save plot to this file instead of showing it
    kwargs: arguments passed to the ScatterBokeh plot
    """
    assert emb.shape[1] >= 2
    if emb.shape[1] > 2:
        from .embeddingutil import embed_2d
        emb = embed_2d(emb, emb_method)

    x, y = emb.T
    ScatterBokeh(x=x, y=y, **kwargs).plot().save_or_show(outfile=outfile)
