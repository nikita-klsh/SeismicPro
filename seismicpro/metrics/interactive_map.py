"""Implements classes for interactive metric map plotting"""

from functools import partial

from sklearn.neighbors import NearestNeighbors

from ..utils import get_text_formatting_kwargs, align_args, MissingModule
from ..utils.interactive_plot_utils import InteractivePlot, PairedPlot, TEXT_LAYOUT, BUTTON_LAYOUT

# Safe import of modules for interactive plotting
try:
    from ipywidgets import widgets
except ImportError:
    widgets = MissingModule("ipywidgets")


class MapCoordsPlot(InteractivePlot):
    """Construct an interactive plot that passes the last click coordinates to a `plot_fn` of each of its views in
    addition to `ax`."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_coords = None

    @property
    def plot_fn(self):
        """callable: plotter of the current view with the last click coordinates passed."""
        if self.current_coords is None:
            return None
        return partial(super().plot_fn, coords=self.current_coords)


class MapBinPlot(MapCoordsPlot):
    """Construct an interactive plot that displays contents of a metric map bin.

    The plot allows selecting an item in the bin using a dropdown widget and iterating over items in both directions
    using arrow buttons.

    Parameters
    ----------
    options : pandas.Series, optional
        A series with metric data in the bin. Its index must store metric coordinates and values - corresponding metric
        values. If not given, an empty plot is created.
    is_lower_better : bool, optional, defaults to True
        Specifies if lower value of the metric is better. Affects the default sorting of bin contents to first display
        a plot for the worst metric value.
    kwargs : misc, optional
        Additional keyword arguments to :func:`~InteractivePlot.__init__`.
    """
    def __init__(self, options=None, is_lower_better=True, **kwargs):
        self.is_desc = is_lower_better
        self.options = None
        self.curr_option = None

        self.sort = widgets.Button(icon=self.sort_icon, disabled=True, layout=widgets.Layout(**BUTTON_LAYOUT))
        self.prev = widgets.Button(icon="angle-left", disabled=True, layout=widgets.Layout(**BUTTON_LAYOUT))
        self.drop = widgets.Dropdown(layout=widgets.Layout(**TEXT_LAYOUT))
        self.next = widgets.Button(icon="angle-right", disabled=True, layout=widgets.Layout(**BUTTON_LAYOUT))

        # Handler definition
        self.sort.on_click(self.reverse_options)
        self.prev.on_click(self.prev_option)
        self.drop.observe(self.select_option, names="value")
        self.next.on_click(self.next_option)

        super().__init__(**kwargs)
        if options is not None:
            self.update_state(0, options)

    def construct_header(self):
        """Construct a header of the plot that contains a dropdown widget with bin contents, metric sort button and
        arrow buttons to iterate over bin items in both directions."""
        return widgets.HBox([self.sort, self.prev, self.drop, self.next])

    @property
    def sort_icon(self):
        """str: current sort icon."""
        return "sort-amount-desc" if self.is_desc else "sort-amount-asc"

    @property
    def drop_options(self):
        """list of str: text representation of bin items."""
        return [f"{metric:.05f} metric at ({x}, {y})" for (x, y), metric in self.options.iteritems()]

    def update_state(self, option_ix, options=None, redraw=True):
        """Set new plot options and the currently active option."""
        new_options = self.options if options is None else options
        if (new_options is None) or (option_ix < 0) or (option_ix >= len(new_options)):
            return
        self.options = new_options
        self.curr_option = option_ix
        self.current_coords = self.options.index[self.curr_option]

        # Unobserve dropdown widget to simultaneously update both options and the currently selected option
        self.drop.unobserve(self.select_option, names="value")
        with self.drop.hold_sync():
            self.drop.options = self.drop_options
            self.drop.index = self.curr_option
        self.drop.observe(self.select_option, names="value")

        self.sort.disabled = False
        self.prev.disabled = (self.curr_option == 0)
        self.next.disabled = (self.curr_option == (len(self.options) - 1))

        if redraw:
            self.redraw()

    def reverse_options(self, event):
        """Reverse order of options in the bin. Keep the currently active item unchanged."""
        _ = event
        self.is_desc = not self.is_desc
        self.sort.icon = self.sort_icon
        self.update_state(len(self.options) - self.curr_option - 1, self.options.iloc[::-1], redraw=False)

    def next_option(self, event):
        """Switch to the next item in the bin."""
        _ = event
        self.update_state(min(self.curr_option + 1, len(self.options) - 1))

    def prev_option(self, event):
        """Switch to the previous item in the bin."""
        _ = event
        self.update_state(max(self.curr_option - 1, 0))

    def select_option(self, change):
        """Select an item in the bin."""
        _ = change
        self.update_state(self.drop.index)


class MetricMapPlot(PairedPlot):  # pylint: disable=abstract-method
    """Base class for interactive metric map visualization.

    Two methods should be redefined in a concrete plotter child class:
    * `construct_aux_plot` - construct an interactive plot with map contents at click location,
    * `click` - handle a click on the map plot.
    """
    def __init__(self, metric_map, plot_on_click=None, plot_on_click_kwargs=None, title=None, is_lower_better=None,
                 figsize=(4.5, 4.5), fontsize=8, orientation="horizontal", **kwargs):
        kwargs = {"fontsize": fontsize, **kwargs}
        text_kwargs = get_text_formatting_kwargs(**kwargs)
        plot_on_click, plot_on_click_kwargs = align_args(plot_on_click, plot_on_click_kwargs)
        plot_on_click_kwargs = [{} if plot_kwargs is None else plot_kwargs for plot_kwargs in plot_on_click_kwargs]
        plot_on_click_kwargs = [{**text_kwargs, **plot_kwargs} for plot_kwargs in plot_on_click_kwargs]

        self.figsize = figsize
        self.orientation = orientation

        self.metric_map = metric_map
        self.title = metric_map.plot_title if title is None else title
        self.plot_map = partial(metric_map.plot, title="", is_lower_better=is_lower_better, **kwargs)
        self.plot_on_click = [partial(plot_fn, **plot_kwargs)
                              for plot_fn, plot_kwargs in zip(plot_on_click, plot_on_click_kwargs)]
        self.init_click_coords = metric_map.get_worst_coords(is_lower_better)
        super().__init__(orientation=orientation)

    def construct_main_plot(self):
        """Construct the metric map plot."""
        return InteractivePlot(plot_fn=self.plot_map, click_fn=self.click, title=self.title, figsize=self.figsize)

    def click(self, coords):
        """Handle a click on the map plot."""
        _ = coords
        raise NotImplementedError

    def plot(self):
        """Display the map and perform initial clicking."""
        super().plot()
        self.main.click(self.init_click_coords)


class ScatterMapPlot(MetricMapPlot):
    """Construct an interactive plot of a non-aggregated metric map."""

    def __init__(self, metric_map, plot_on_click, **kwargs):
        self.coords = metric_map.map_data.index.to_frame().values
        self.coords_neighbors = NearestNeighbors(n_neighbors=1).fit(self.coords)
        super().__init__(metric_map, plot_on_click, **kwargs)

    def aux_title(self):
        """Return the title of the map data plot."""
        coords = self.aux.current_coords
        if coords is None:
            return ""
        return f"{self.metric_map.map_data[coords]:.05f} metric at {coords}"

    def construct_aux_plot(self):
        """Construct an interactive plot with data representation at click location."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        return MapCoordsPlot(plot_fn=self.plot_on_click, title=self.aux_title, toolbar_position=toolbar_position)

    def click(self, coords):
        """Get map coordinates closest to click `coords` and draw their data representation."""
        coords_ix = self.coords_neighbors.kneighbors([coords], return_distance=False).item()
        coords = tuple(self.coords[coords_ix])
        self.aux.current_coords = coords
        self.aux.redraw()
        return coords


class BinarizedMapPlot(MetricMapPlot):
    """Construct an interactive plot of a binarized metric map."""

    def construct_aux_plot(self):
        """Construct an interactive plot with map contents at click location."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        return MapBinPlot(plot_fn=self.plot_on_click, toolbar_position=toolbar_position)

    def click(self, coords):
        """Get contents of a map bin by its `coords` and set them as options of the bin contents plot."""
        bin_coords = (int(coords[0] + 0.5), int(coords[1] + 0.5))
        contents = self.metric_map.get_bin_contents(bin_coords)
        if contents is None:  # Handle clicks outside bins
            return None
        self.aux.update_state(0, contents)
        return bin_coords
