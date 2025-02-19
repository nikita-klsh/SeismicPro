"""Building blocks for interactive plots"""

from time import time
from functools import partial

import matplotlib.pyplot as plt

from .general_utils import to_list, get_first_defined, align_args, MissingModule

# Safe import of modules for interactive plotting
try:
    from ipywidgets import widgets
except ImportError:
    widgets = MissingModule("ipywidgets")

try:
    from IPython.display import display
except ImportError:
    display = MissingModule("IPython.display")


# Maximum time between mouse button click and release events to consider them as a single click
MAX_CLICK_TIME = 0.2


# Default widget height
WIDGET_HEIGHT = "30px"


# Default button widgets layout
BUTTON_LAYOUT = {
    "height": WIDGET_HEIGHT,
    "width": "40px",
    "min_width": "40px",
}


# HTML style of plot titles
TITLE_STYLE = "<style>p{word-wrap:normal; text-align:center; font-size:14px}</style>"
TITLE_TEMPLATE = "{style} <b><p>{title}</p></b>"


class InteractivePlot:  # pylint: disable=too-many-instance-attributes
    """Construct an interactive plot with optional click handling.

    The plot may contain multiple views: one for each of the passed `plot_fn`. If more than one view is defined, an
    extra button is created in the toolbar to iterate over them. The plot is interactive: it can handle click and slice
    events, while each view may define its own processing logic.

    Plotting must be performed in a JupyterLab environment with the `%matplotlib widget` magic executed and `ipympl`
    and `ipywidgets` libraries installed.

    Parameters
    ----------
    plot_fn : callable or list of callable, optional
        One or more plotters each accepting a single keyword argument `ax`. If more than one plotter is given, an extra
        button for view switching is displayed. If not given, an empty plot is created.
    click_fn : callable or list of callable, optional
        Click handlers for views defined by `plot_fn`. Each of them must accept a tuple with 2 elements defining
        click coordinates. If a single `click_fn` is given, it is used for all views. If not given, click events are
        not handled.
    slice_fn : callable or list of callable, optional
        Slice handlers for views defined by `plot_fn`. Slice is triggered by moving the mouse with the left button
        held. Each handlers must accept two tuples with 2 elements defining coordinates of slice edges. If a single
        `slice_fn` is given, it is used for all views. If not given, slice events are not handled.
    unclick_fn : callable or list of callable, optional
        Handlers that undo clicks and slices on views defined by `plot_fn`. Each of them is called without arguments.
        If a single `unclick_fn` is given, it is used for all views. If not given, clicks and slices can not be undone.
    marker_params : dict or list of dict, optional, defaults to {"marker": "+", "color": "black"}
        Click marker parameters for views defined by `plot_fn`. Passed directly to `Axes.scatter`. If a single `dict`
        is given, it is used for all views.
    title : str or callable or list of str or callable, optional
        Plot titles for views defined by `plot_fn`. If `callable`, it is called each time the title is being set (e.g.
        on `redraw`) allowing for dynamic title generation. If not given, an empty title is created.
    preserve_clicks_on_view_change : bool, optional, defaults to False
        Whether to preserve click/slice markers and trigger the corresponding event on view change.
    preserve_lims : bool, optional, defaults to False
        Whether to preserve limits changes on each views on its redraw. If `self.plot_fn` is not given, limits won't be
        preserved.
    preserve_lims_on_view_change : bool, optional, defaults to False
        Whether to preserve limits changes on view change.
    toolbar_position : {"top", "bottom", "left", "right"}, optional, defaults to "left"
        Toolbar position relative to the main axes.
    figsize : tuple with 2 elements, optional, defaults to (4.5, 4.5)
        Size of the created figure. Measured in inches.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        Axes of the figure to plot views on.
    box : ipywidgets.widgets.widget_box.Box
        Main container that stores figure canvas, plot title, created buttons and, optionally, a toolbar.
    n_views : int
        The number of plot views.
    current_view : int
        An index of the current plot view.
    """
    # pylint: disable-next=too-many-arguments, too-many-statements
    def __init__(self, *, plot_fn=None, click_fn=None, slice_fn=None, unclick_fn=None, marker_params=None, title="",
                 preserve_clicks_on_view_change=False, preserve_lims=False, preserve_lims_on_view_change=False,
                 toolbar_position="left", figsize=(4.5, 4.5)):
        if "ipympl" not in plt.get_backend():
            raise RuntimeError("Plotting must be performed in a JupyterLab environment "
                               "with the `%matplotlib widget` magic executed")

        list_args = align_args(plot_fn, click_fn, slice_fn, unclick_fn, marker_params, title)
        self.plot_fn_list, self.click_fn_list, self.slice_fn_list = list_args[:3]
        self.unclick_fn_list, marker_params_list, self.title_list = list_args[3:]
        self.marker_params_list = []
        for params in marker_params_list:
            if params is None:
                params = {}
            params = {"marker": "+", "color": "black", **params}
            self.marker_params_list.append(params)

        # View-related attributes
        self.n_views = len(self.plot_fn_list)
        self.current_view = 0
        self.preserve_clicks_on_view_change = preserve_clicks_on_view_change

        # Preserve limits-related attributes
        self.preserve_lims = preserve_lims
        self.preserve_lims_on_view_change = preserve_lims_on_view_change
        self.current_axes_lims = None
        self.home_axes_lims = None
        self._axes_callbacks_oids = []

        # Click-related attributes
        self.start_click_time = None
        self.start_click_coords = None
        self.click_coords = None
        self.slice_coords = None
        self.click_marker = None
        self.slice_marker = None

        # Construct a figure
        with plt.ioff():
            # Add tight_layout to always correctly show colorbar ticks
            self.fig, self.ax = plt.subplots(figsize=figsize, tight_layout=True)  # pylint: disable=invalid-name
        self.fig.canvas.header_visible = False
        self.fig.canvas.toolbar_visible = False

        # Setup event handlers
        self.fig.interactive_plotter = self  # Always keep reference to self for all plots to remain interactive
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)

        # Define widgets and toolbar buttons
        # Non-toggle buttons have a one-space description to be aligned with toggle buttons, which append it even if
        # only icon is defined, keep it as is until https://github.com/jupyter-widgets/ipywidgets/issues/2209 is fixed
        self.title_widget = widgets.HTML(value="", layout=widgets.Layout(height=WIDGET_HEIGHT))
        self.view_button = widgets.Button(icon="exchange", tooltip="Switch to the next view", description=" ",
                                          layout=widgets.Layout(**BUTTON_LAYOUT))
        self.view_button.on_click(self.on_view_toggle)
        self.home_button = widgets.Button(icon="home", tooltip="Reset original view", description=" ",
                                          layout=widgets.Layout(**BUTTON_LAYOUT))
        self.home_button.on_click(self.on_home_toggle)
        self.back_button = widgets.Button(icon="arrow-left", tooltip="Back to previous view", description=" ",
                                          layout=widgets.Layout(**BUTTON_LAYOUT))
        self.back_button.on_click(self.fig.canvas.toolbar.back)
        self.forward_button = widgets.Button(icon="arrow-right", tooltip="Forward to next view", description=" ",
                                             layout=widgets.Layout(**BUTTON_LAYOUT))
        self.forward_button.on_click(self.fig.canvas.toolbar.forward)
        self.pan_button = widgets.ToggleButton(icon="arrows", tooltip="Move the plot",
                                               layout=widgets.Layout(**BUTTON_LAYOUT))
        self.pan_button.observe(self.on_pan_toggle, "value")
        self.zoom_button = widgets.ToggleButton(icon="square-o", tooltip="Zoom to rectangle",
                                                layout=widgets.Layout(**BUTTON_LAYOUT))
        self.zoom_button.observe(self.on_zoom_toggle, "value")
        self.save_button = widgets.Button(icon="save", tooltip="Download plot", description=" ",
                                          layout=widgets.Layout(**BUTTON_LAYOUT))
        self.save_button.on_click(self.fig.canvas.toolbar.save_figure)

        # Build plot box
        available_positions = {"top", "bottom", "left", "right"}
        if toolbar_position not in available_positions:
            raise ValueError(f"Unknown toolbar position, must be one of {', '.join(available_positions)}")
        self.toolbar_position = toolbar_position
        self.header = self.construct_header()
        self.toolbar = self.construct_toolbar()
        self.box = self.construct_box()

    def __del__(self):
        """Close the figure on plot deletion."""
        del self.fig.interactive_plotter
        plt.close(self.fig)

    @property
    def plot_fn(self):
        """callable: plotter of the current view."""
        return self.plot_fn_list[self.current_view]

    @property
    def click_fn(self):
        """callable: click handler of the current view."""
        return self.click_fn_list[self.current_view]

    @property
    def is_clickable(self):
        """bool: whether the current view is clickable."""
        return self.click_fn is not None

    @property
    def slice_fn(self):
        """callable: slice handler of the current view."""
        return self.slice_fn_list[self.current_view]

    @property
    def is_sliceable(self):
        """bool: whether the current view is sliceable."""
        return self.slice_fn is not None

    @property
    def unclick_fn(self):
        """callable: undo click or slice on the current view."""
        return self.unclick_fn_list[self.current_view]

    @property
    def is_unclickable(self):
        """bool: whether the click/slice can be undone for the current view."""
        return self.unclick_fn is not None

    @property
    def marker_params(self):
        """dict: click marker parameters of the current view."""
        return self.marker_params_list[self.current_view]

    @property
    def title(self):
        """str: title of the current view. Evaluates callable titles."""
        title = self.title_list[self.current_view]
        if callable(title):
            return title()
        return title

    # Box construction

    def construct_header(self):
        """Construct a header of the plot containing the view title."""
        return self.title_widget

    def construct_extra_buttons(self):
        """Return a list of extra buttons to add to a toolbar. Can be overridden in child classes."""
        if self.n_views == 1:
            return []
        return [self.view_button]

    def construct_toolbar(self):
        """Construct a plot toolbar which contains the toolbar of the canvas and constructed buttons."""
        box_type = widgets.HBox if self.toolbar_position in {"top", "bottom"} else widgets.VBox
        toolbar_buttons = [self.home_button, self.back_button, self.forward_button, self.pan_button, self.zoom_button,
                           self.save_button]
        return box_type(self.construct_extra_buttons() + toolbar_buttons)

    def construct_box(self):
        """Construct the box of the whole plot which contains figure canvas, header and a toolbar."""
        titled_box = widgets.HBox([widgets.VBox([self.header, self.fig.canvas])])
        return attach_widget(titled_box, self.toolbar, position=self.toolbar_position)

    # Event handlers

    def on_resize(self, event):
        """Resize the plot on the `fig` canvas size change."""
        self.resize(event.width)

    def on_click(self, event):
        """Remember the mouse button click time to further distinguish between mouse click and hold events."""
        if event.inaxes != self.ax:
            return  # Discard clicks outside the main axes
        if event.button == 1:
            self.start_click_time = time()
            self.start_click_coords = (event.xdata, event.ydata)

    def on_motion(self, event):
        """Handle mouse movement with the pressed left mouse button. Redraw currently selected slice line."""
        if self.is_sliceable and event.button == 1 and not self.pan_button.value and not self.zoom_button.value:
            self._plot_slice(self.start_click_coords, (event.xdata, event.ydata))

    def on_release(self, event):
        """Handle clicks and slices of the plot."""
        if event.inaxes != self.ax:
            return  # Discard clicks outside the main axes
        if time() - self.start_click_time < MAX_CLICK_TIME:  # Single click
            if (event.inaxes == self.ax) and (event.button == 1):
                self.click((event.xdata, event.ydata))
            elif self.click_coords is not None:  # Restore previous valid click
                self.click(self.click_coords)
        elif not self.pan_button.value and not self.zoom_button.value:
            # Process slice only if "Zoom" or "Pad" modes are not selected
            if (event.inaxes == self.ax) and (event.button == 1):
                self.slice(self.start_click_coords, (event.xdata, event.ydata))
            elif self.slice_coords is not None:  # Restore previous valid slice
                self.slice(*self.slice_coords)
        self.start_click_time = None
        self.start_click_coords = None

    def on_press(self, event):
        """Undo mouse click or slice on ESC key press if allowed."""
        if (event.inaxes == self.ax) and (event.key == "escape"):
            self.unclick()

    def on_view_toggle(self, event):
        """Switch the plot to the next view."""
        _ = event
        self.set_view((self.current_view + 1) % self.n_views)

    def on_home_toggle(self, event):
        """Toggle home button."""
        _ = event
        self.fig.canvas.toolbar.home()
        # Manually set original axes limits since toolbar won't reset them after axes redraw
        if self.home_axes_lims is not None:
            self.ax.set_xlim(self.home_axes_lims[0])
            self.ax.set_ylim(self.home_axes_lims[1])
        self.current_axes_lims = None

    def on_pan_toggle(self, event):
        """Toggle pan button."""
        _ = event
        if self.zoom_button.value:
            self.zoom_button.unobserve_all()  # Avoid recursion during value setting
            self.zoom_button.value = False
            self.zoom_button.observe(self.on_zoom_toggle, "value")
        self.fig.canvas.toolbar.pan()

    def on_zoom_toggle(self, event):
        """Toggle zoom button."""
        _ = event
        if self.pan_button.value:
            self.pan_button.unobserve_all()  # Avoid recursion during value setting
            self.pan_button.value = False
            self.pan_button.observe(self.on_pan_toggle, "value")
        self.fig.canvas.toolbar.zoom()

    # Axes callbacks

    def set_axes_callbacks(self):
        """Set axes callbacks."""
        xlim_oid = self.ax.callbacks.connect("xlim_changed", self.update_lims)
        ylim_oid = self.ax.callbacks.connect("ylim_changed", self.update_lims)
        self._axes_callbacks_oids.extend([xlim_oid, ylim_oid])

    def update_lims(self, ax):
        """Update `current_axes_lims` on limits change."""
        self.current_axes_lims = (ax.get_xlim(), ax.get_ylim())

    def remove_axes_callbacks(self):
        """Remove all axes callbacks."""
        for oid in self._axes_callbacks_oids:
            self.ax.callbacks.disconnect(oid)
        self._axes_callbacks_oids = []

    # General plot API

    def resize(self, width):
        """Resize the plot to have the given `width`."""
        width += 4  # Correction for main axes margins
        self.header.layout.width = f"{int(width)}px"

    def _clear_markers(self):
        """Remove click and slice markers. Does not force figure redrawing."""
        if self.click_marker is not None:
            self.click_marker.remove()
        self.click_marker = None

        if self.slice_marker is not None:
            self.slice_marker.remove()
        self.slice_marker = None

    def click(self, coords):
        """Trigger a click on the plot at given `coords`."""
        if not self.is_clickable:
            return
        coords = self.click_fn(coords)
        if coords is None:  # Ignore click
            return
        self._clear_markers()
        self.start_click_coords = None
        self.click_coords = coords
        self.slice_coords = None
        self.click_marker = self.ax.scatter(*coords, **self.marker_params, zorder=10)
        self.fig.canvas.draw_idle()

    def _plot_slice(self, start_coords, stop_coords):
        """Plot a line segment from `start_coords` to `stop_coords`."""
        self._clear_markers()
        self.slice_marker = self.ax.plot([start_coords[0], stop_coords[0]], [start_coords[1], stop_coords[1]],
                                         color="black", zorder=10)[0]
        self.fig.canvas.draw_idle()

    def slice(self, start_coords, stop_coords):
        """Trigger slicing of the plot from `start_coords` to `stop_coords`."""
        if not self.is_sliceable:
            return
        self.slice_fn(start_coords, stop_coords)
        self.start_click_coords = None
        self.click_coords = None
        self.slice_coords = (start_coords, stop_coords)
        self._plot_slice(start_coords, stop_coords)

    def unclick(self):
        """Undo last click or slice event."""
        if not self.is_unclickable:
            return
        if self.click_marker is None and self.slice_marker is None:
            return  # Do nothing if a click has not been performed
        self.unclick_fn()
        self._clear_markers()
        self.fig.canvas.draw_idle()

    def set_view(self, view):
        """Set the current view of the plot to the given `view`."""
        if view < 0 or view >= self.n_views:
            raise ValueError("Unknown view")
        self.unclick()
        self.current_view = view
        if not self.preserve_clicks_on_view_change:
            self.click_coords = None
            self.slice_coords = None
        self.redraw(preserve_lims=self.preserve_lims_on_view_change)

    def set_title(self, title=None):
        """Update the plot title. If `title` is not given, the default title of the current view is used."""
        title = get_first_defined(title, self.title)
        self.title_widget.value = TITLE_TEMPLATE.format(style=TITLE_STYLE, title=title)

    def clear(self):
        """Clear the plot axes and revert them to the initial state."""
        # Remove callbacks to avoid its trigger on empty axes
        self.remove_axes_callbacks()
        self.home_axes_lims = None
        # Remove all axes except for the main one if they were created (e.g. a colorbar)
        for ax in self.fig.axes:
            if ax != self.ax:
                ax.remove()
        self.ax.clear()
        # Reset aspect ratio constraints if they were set
        self.ax.set_aspect("auto")
        # Stretch the axes to its original size
        self.ax.set_axes_locator(None)
        # Reset toolbar buttons history
        self.fig.canvas.toolbar.update()

    def redraw(self, clear=True, preserve_lims=None):
        """Redraw the current view. Optionally clear the plot axes first."""
        if clear:
            self.clear()
        self.set_title()
        if self.plot_fn is not None:
            self.plot_fn(ax=self.ax)  # pylint: disable=not-callable
            # Save the current axes limits to be able to restore them on a home button toggle
            self.home_axes_lims = (self.ax.get_xlim(), self.ax.get_ylim())
            if preserve_lims is None:
                preserve_lims = self.preserve_lims
            if not preserve_lims:
                self.current_axes_lims = None
            if self.current_axes_lims is not None:
                self.ax.set_xlim(self.current_axes_lims[0])
                self.ax.set_ylim(self.current_axes_lims[1])
            self.set_axes_callbacks()
        if self.click_coords is not None:
            self.click(self.click_coords)
        if self.slice_coords is not None:
            self.slice(*self.slice_coords)

    def plot(self, display_box=True):
        """Display the interactive plot with the first view selected.

        Parameters
        ----------
        display_box : bool, optional, defaults to True
            Whether to display the plot in a JupyterLab frontend. Generally should be set to `False` if a parent object
            creates several `InteractivePlot` instances and controls their plotting.
        """
        self.redraw(clear=False)
        # Init the width of the box
        self.resize(self.fig.get_figwidth() * self.fig.dpi / self.fig.canvas.device_pixel_ratio)
        if display_box:
            display(self.box)


class DropdownViewPlot(InteractivePlot):
    """Construct an interactive plot with optional click handling.

    The plot may contain multiple views: one for each of the passed `plot_fn`. The views can be iterated over either
    using a dropdown list on top of the plot or arrow buttons on its sides. The plot is interactive: it can handle
    click and slice events, while each view may define its own processing logic.

    Plotting must be performed in a JupyterLab environment with the `%matplotlib widget` magic executed and `ipympl`
    and `ipywidgets` libraries installed.

    Parameters
    ----------
    plot_fn : callable or list of callable, optional
        One or more plotters each accepting a single keyword argument `ax`. If not given, an empty plot is created.
    click_fn : callable or list of callable, optional
        Click handlers for views defined by `plot_fn`. Each of them must accept a tuple with 2 elements defining
        click coordinates. If a single `click_fn` is given, it is used for all views. If not given, click events are
        not handled.
    slice_fn : callable or list of callable, optional
        Slice handlers for views defined by `plot_fn`. Slice is triggered by moving the mouse with the left button
        held. Each handlers must accept two tuples with 2 elements defining coordinates of slice edges. If a single
        `slice_fn` is given, it is used for all views. If not given, slice events are not handled.
    unclick_fn : callable or list of callable, optional
        Handlers that undo clicks and slices on views defined by `plot_fn`. Each of them is called without arguments.
        If a single `unclick_fn` is given, it is used for all views. If not given, clicks and slices can not be undone.
    marker_params : dict or list of dict, optional, defaults to {"marker": "+", "color": "black"}
        Click marker parameters for views defined by `plot_fn`. Passed directly to `Axes.scatter`. If a single `dict`
        is given, it is used for all views.
    title : str or list of str, optional
        Plot titles for views defined by `plot_fn`, act as dropdown options.
    preserve_clicks_on_view_change : bool, optional, defaults to False
        Whether to preserve click/slice markers and trigger the corresponding event on view change.
    preserve_lims : bool, optional, defaults to False
        Whether to preserve limits changes on each views on its redraw. If `self.plot_fn` is not given, limits won't be
        preserved.
    preserve_lims_on_view_change : bool, optional, defaults to False
        Whether to preserve limits changes on view change.
    toolbar_position : {"top", "bottom", "left", "right"}, optional, defaults to "left"
        Toolbar position relative to the main axes.
    figsize : tuple with 2 elements, optional, defaults to (4.5, 4.5)
        Size of the created figure. Measured in inches.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        Axes of the figure to plot views on.
    box : ipywidgets.widgets.widget_box.Box
        Main container that stores figure canvas, plot title, created buttons and, optionally, a toolbar.
    n_views : int
        The number of plot views.
    current_view : int
        An index of the current plot view.
    """
    def __init__(self, **kwargs):
        # Define widgets for view selection
        self.prev = widgets.Button(icon="angle-left", tooltip="", disabled=True,
                                   layout=widgets.Layout(**BUTTON_LAYOUT))
        self.drop = widgets.Dropdown(layout=widgets.Layout(height=WIDGET_HEIGHT, width="inherit"))
        self.next = widgets.Button(icon="angle-right", tooltip="", disabled=True,
                                   layout=widgets.Layout(**BUTTON_LAYOUT))

        super().__init__(**kwargs)
        self.drop.options = self.title_list
        self.drop.index = 0

        # Define handlers after options are set, otherwise plotting will be triggered
        self.prev.on_click(self.prev_view)
        self.drop.observe(self.select_view, names="value")
        self.next.on_click(self.next_view)

    def construct_extra_buttons(self):
        """Don't use a parent button for view switching."""
        return []

    def construct_header(self):
        """Construct a header of the plot. Contains a dropdown widget with available views and two arrow buttons on its
        sides to iterate over views."""
        return widgets.HBox([self.prev, self.drop, self.next])

    def set_view(self, view):
        """Set the current view of the plot to the given `view`."""
        super().set_view(view)
        self.drop.index = view
        self.prev.disabled = view == 0
        self.next.disabled = view == (self.n_views - 1)

    def next_view(self, event):
        """Switch to the next view."""
        _ = event
        self.set_view(min(self.current_view + 1, self.n_views - 1))

    def prev_view(self, event):
        """Switch to the previous view."""
        _ = event
        self.set_view(max(self.current_view - 1, 0))

    def select_view(self, change):
        """Set the current view of the plot according to the selected dropdown option."""
        _ = change
        self.set_view(self.drop.index)


class DropdownOptionPlot(InteractivePlot):
    """Construct an interactive plot that changes the behavior of `plot_fn` depending on the chosen option: each of
    them defines its own keyword arguments passed to the current view plotter in addition to `ax`.

    The plot allows selecting an option using a dropdown widget and iterating over options in both directions using
    arrow buttons.

    Parameters
    ----------
    options : list of dict, optional
        Available options. All options must have the same keys. `option_title` is an obligatory key, that defines
        displayed label of the option in the dropdown widget. All other parameters are passed to the current view
        plotter in addition to `ax`.
    args, kwargs : misc, optional
        Additional arguments to :func:`~InteractivePlot.__init__`.
    """
    def __init__(self, *args, options=None, **kwargs):
        # Define widgets for option selection
        self.sort = widgets.Button(icon="sort", tooltip="", disabled=True,
                                   layout=widgets.Layout(**BUTTON_LAYOUT))
        self.prev = widgets.Button(icon="angle-left", tooltip="", disabled=True,
                                   layout=widgets.Layout(**BUTTON_LAYOUT))
        self.drop = widgets.Dropdown(layout=widgets.Layout(height=WIDGET_HEIGHT, width="inherit"))
        self.next = widgets.Button(icon="angle-right", tooltip="", disabled=True,
                                   layout=widgets.Layout(**BUTTON_LAYOUT))

        # Define handlers
        self.sort.on_click(self.reverse_options)
        self.prev.on_click(self.prev_option)
        self.drop.observe(self.select_option, names="value")
        self.next.on_click(self.next_option)

        super().__init__(*args, **kwargs)

        self.options = None
        self.current_option_ix = None
        if options is not None:
            self.update_state(0, options)

    def construct_header(self):
        """Construct a header of the plot that contains a dropdown widget with available options, sorting button and
        arrow buttons to iterate over options in both directions."""
        return widgets.HBox([self.sort, self.prev, self.drop, self.next])

    @property
    def plot_fn(self):
        """callable: Plotter of the current view with passed parameters of the selected option."""
        if self.options is None:
            return None
        plot_kwargs = {key: val for key, val in self.options[self.current_option_ix].items() if key != "option_title"}
        return partial(super().plot_fn, **plot_kwargs)

    def update_state(self, option_ix, options=None, redraw=True):
        """Set new plot options and the currently active option."""
        new_options = self.options
        if options is not None:
            if not isinstance(options, (list, tuple)) or not all(isinstance(option, dict) for option in options):
                raise TypeError("options must be a list or tuple of dicts")
            if any(option.keys() != options[0].keys() for option in options):
                raise KeyError("All options must have the same keys")
            if "option_title" not in options[0].keys():
                raise KeyError("All options must have a title")
            new_options = options
        if (new_options is None) or (option_ix < 0) or (option_ix >= len(new_options)):
            return

        self.options = new_options
        self.current_option_ix = option_ix

        # Unobserve dropdown widget to simultaneously update both options and the currently selected option
        self.drop.unobserve(self.select_option, names="value")
        with self.drop.hold_sync():
            self.drop.options = [option["option_title"] for option in new_options]
            self.drop.index = self.current_option_ix
        self.drop.observe(self.select_option, names="value")

        self.sort.disabled = False
        self.prev.disabled = self.current_option_ix == 0
        self.next.disabled = self.current_option_ix == (len(self.options) - 1)

        if redraw:
            self.redraw(preserve_lims=self.preserve_lims_on_view_change)

    def reverse_options(self, event):
        """Reverse options order. Keep the currently active option unchanged."""
        _ = event
        self.update_state(len(self.options) - self.current_option_ix - 1, self.options[::-1], redraw=False)

    def next_option(self, event):
        """Switch to the next option."""
        _ = event
        self.update_state(min(self.current_option_ix + 1, len(self.options) - 1))

    def prev_option(self, event):
        """Switch to the previous option."""
        _ = event
        self.update_state(max(self.current_option_ix - 1, 0))

    def select_option(self, change):
        """Select an option."""
        _ = change
        self.update_state(self.drop.index)


class ToggleButtonsPlot(InteractivePlot):
    """Construct an interactive plot with optional click handling.

    The plot may contain multiple views: one for each of the passed `plot_fn`. A toggle button is created in the
    toolbar for each view. The plot is interactive: it can handle click and slice events, while each view may define
    its own processing logic.

    Plotting must be performed in a JupyterLab environment with the `%matplotlib widget` magic executed and `ipympl`
    and `ipywidgets` libraries installed.

    Parameters
    ----------
    plot_fn : callable or list of callable, optional
        One or more plotters each accepting a single keyword argument `ax`. If not given, an empty plot is created.
    names : src or list of str or None, optional
        List of descriptions displayed on the toggle buttons. If None, numbers started from 1 to n_views will be used.
        If `icons` is not None, this parameter will be omitted.
    icons : src or list of str or None, optional
        List of font-awesome icon names for each toggle button.
    buttons_position : {"top", "bottom", "left", "right"}, optional, defaults to "right"
        Toggle buttons position relative to the toolbar.
    click_fn : callable or list of callable, optional
        Click handlers for views defined by `plot_fn`. Each of them must accept a tuple with 2 elements defining
        click coordinates. If a single `click_fn` is given, it is used for all views. If not given, click events are
        not handled.
    slice_fn : callable or list of callable, optional
        Slice handlers for views defined by `plot_fn`. Slice is triggered by moving the mouse with the left button
        held. Each handlers must accept two tuples with 2 elements defining coordinates of slice edges. If a single
        `slice_fn` is given, it is used for all views. If not given, slice events are not handled.
    unclick_fn : callable or list of callable, optional
        Handlers that undo clicks and slices on views defined by `plot_fn`. Each of them is called without arguments.
        If a single `unclick_fn` is given, it is used for all views. If not given, clicks and slices can not be undone.
    marker_params : dict or list of dict, optional, defaults to {"marker": "+", "color": "black"}
        Click marker parameters for views defined by `plot_fn`. Passed directly to `Axes.scatter`. If a single `dict`
        is given, it is used for all views.
    title : str or callable or list of str or callable, optional
        Plot titles for views defined by `plot_fn`. If `callable`, it is called each time the title is being set (e.g.
        on `redraw`) allowing for dynamic title generation. If not given, an empty title is created.
    preserve_clicks_on_view_change : bool, optional, defaults to False
        Whether to preserve click/slice markers and trigger the corresponding event on view change.
    preserve_lims : bool, optional, defaults to False
        Whether to preserve limits changes on each views on its redraw. If `self.plot_fn` is not given, limits won't be
        preserved.
    preserve_lims_on_view_change : bool, optional, defaults to False
        Whether to preserve limits changes on view change.
    toolbar_position : {"top", "bottom", "left", "right"}, optional, defaults to "left"
        Toolbar position relative to the main axes.
    figsize : tuple with 2 elements, optional, defaults to (4.5, 4.5)
        Size of the created figure. Measured in inches.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        Axes of the figure to plot views on.
    box : ipywidgets.widgets.widget_box.Box
        Main container that stores figure canvas, plot title, created toggle buttons and a toolbar.
    n_views : int
        The number of plot views.
    current_view : int
        An index of the current plot view.
    view_toggle_buttons : list
        A list with toggle buttons responsible for view change.
    """
    def __init__(self, *, plot_fn=None, names=None, icons=None, buttons_position="right", **kwargs):
        plot_fn_list = to_list(plot_fn)
        if icons is not None:
            icons = to_list(names)
            if len(icons) != len(plot_fn_list):
                raise ValueError("The length of `icons` must match number of views")
            buttons_kwargs = [{"icon": icon} for icon in icons]
        elif names is not None:
            names = to_list(names)
            if len(names) != len(plot_fn_list):
                raise ValueError("The length of `names` must match number of views")
            buttons_kwargs = [{"description": name} for name in names]
        else:
            buttons_kwargs = [{"description": str(i+1)} for i, _ in enumerate(plot_fn_list)]

        # Define widget buttons
        self.view_toggle_buttons = []
        for button_kwargs in buttons_kwargs:
            button = widgets.ToggleButton(layout=widgets.Layout(**{**BUTTON_LAYOUT, "width": "auto"}), **button_kwargs)
            button.observe(self.on_button_toggle, "value")
            self.view_toggle_buttons.append(button)

        available_buttons_positions = {"top", "bottom", "left", "right"}
        if buttons_position not in available_buttons_positions:
            raise ValueError(f"Unknown buttons position, must be one of {', '.join(available_buttons_positions)}")
        self.buttons_position = buttons_position

        super().__init__(plot_fn=plot_fn, **kwargs)

    def construct_extra_buttons(self):
        """Don't use a parent button for view switching."""
        return []

    def construct_toolbar(self):
        """Construct a plot toolbar and attach toggle buttons to it."""
        toolbar = super().construct_toolbar()
        button_box_type = widgets.HBox if self.toolbar_position in {"top", "bottom"} else widgets.VBox
        buttons = button_box_type(self.view_toggle_buttons)
        return attach_widget(toolbar, buttons, position=self.buttons_position)

    def on_button_toggle(self, event):
        """Switch the plot to the view corresponding to the pressed button."""
        pressed_button = event["owner"]
        pressed_button.disabled = True  # Disable pressed button to avoid multiple clicks to the same view
        for ix, button in enumerate(self.view_toggle_buttons):
            if button is pressed_button:
                self.set_view(ix)
            else:  # Disable if button is not pressed
                button.unobserve_all()  # Avoid recursion during value setting
                button.value = False
                button.disabled = False
                button.observe(self.on_button_toggle, "value")


class SlidingPlot(InteractivePlot):
    """Construct an interactive plot with FloatSlider and optional click handling.

    A FloatSlider is located between the header and canvas of the plot. The limits of the slider can be changed either
    with `__init__` or `self.set_slider`.

    The plot may contain multiple views: one for each of the passed `plot_fn`. If more than one view is defined, an
    extra button is created in the toolbar to iterate over them. The plot is interactive: it can handle click and slice
    events, while each view may define its own processing logic.

    Plotting must be performed in a JupyterLab environment with the `%matplotlib widget` magic executed and `ipympl`
    and `ipywidgets` libraries installed.

    Parameters
    ----------
    slider_min : int or float
        Minimal position of the slider.
    slider_max : int or float
        Maximal position of the slider.
    slider_init : int, float or None, optional
        Initial position of the slider. If None, the initial position will be set to the minimal position.
    slider_step : int, float, optional
        Step of the trackbar.
    slide_fn : callable, optional
        Handler is triggered on widgets.FloatSlider move.
    reset_fn : callable, optional
        Button handler to reset the widgets.FloatSlider to its initial position. If not provided, the slider will be
        set to `slider_init` position and the axis will be redrawn.
    slider_kwargs : dict, optional
        Additional arguments for the widgets.FloatSlider.
    plot_fn : callable or list of callable, optional
        One or more plotters each accepting a single keyword argument `ax`. If more than one plotter is given, an extra
        button for view switching is displayed. If not given, an empty plot is created.
    click_fn : callable or list of callable, optional
        Click handlers for views defined by `plot_fn`. Each of them must accept a tuple with 2 elements defining
        click coordinates. If a single `click_fn` is given, it is used for all views. If not given, click events are
        not handled.
    slice_fn : callable or list of callable, optional
        Slice handlers for views defined by `plot_fn`. Slice is triggered by moving the mouse with the left button
        held. Each handlers must accept two tuples with 2 elements defining coordinates of slice edges. If a single
        `slice_fn` is given, it is used for all views. If not given, slice events are not handled.
    unclick_fn : callable or list of callable, optional
        Handlers that undo clicks and slices on views defined by `plot_fn`. Each of them is called without arguments.
        If a single `unclick_fn` is given, it is used for all views. If not given, clicks and slices can not be undone.
    marker_params : dict or list of dict, optional, defaults to {"marker": "+", "color": "black"}
        Click marker parameters for views defined by `plot_fn`. Passed directly to `Axes.scatter`. If a single `dict`
        is given, it is used for all views.
    title : str or callable or list of str or callable, optional
        Plot titles for views defined by `plot_fn`. If `callable`, it is called each time the title is being set (e.g.
        on `redraw`) allowing for dynamic title generation. If not given, an empty title is created.
    preserve_clicks_on_view_change : bool, optional, defaults to False
        Whether to preserve click/slice markers and trigger the corresponding event on view change.
    preserve_lims : bool, optional, defaults to False
        Whether to preserve limits changes on each views on its redraw. If `self.plot_fn` is not given, limits won't be
        preserved.
    preserve_lims_on_view_change : bool, optional, defaults to False
        Whether to preserve limits changes on view change.
    toolbar_position : {"top", "bottom", "left", "right"}, optional, defaults to "left"
        Toolbar position relative to the main axes.
    figsize : tuple with 2 elements, optional, defaults to (4.5, 4.5)
        Size of the created figure. Measured in inches.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        Axes of the figure to plot views on.
    box : ipywidgets.widgets.widget_box.Box
        Main container that stores figure canvas, plot title, created buttons and, optionally, a toolbar.
    n_views : int
        The number of plot views.
    current_view : int
        An index of the current plot view.
    slider_init : int or float
        Initial position of the slider. One can use it in `self.reset_fn` to reset slider to the initial position.
    """
    def __init__(self, *, slider_min, slider_max, slider_init=None, slider_step=None, slide_fn=None, reset_fn=None,
                 slider_kwargs=None, **kwargs):
        self.slide_fn = slide_fn
        self.reset_fn = reset_fn
        self.slider_init = slider_min if slider_init is None else slider_init

        default_slider_kwargs = {
            "readout": False,
            "layout": widgets.Layout(flex="1 1 auto", height=WIDGET_HEIGHT)
        }
        slider_params = {"value": slider_init, "min": slider_min, "max": slider_max, "step": slider_step}
        slider_kwargs = slider_kwargs if slider_kwargs is not None else {}
        self.slider = widgets.FloatSlider(**{**default_slider_kwargs, **slider_kwargs, **slider_params})
        self.slider.observe(self.on_slider_change, "value")
        self.reset_button = widgets.Button(icon="undo", tooltip="Reset to default value",
                                           layout=widgets.Layout(**BUTTON_LAYOUT))
        self.reset_button.on_click(self.on_reset)
        self.min_widget = widgets.HTML(value=self._to_string(slider_min), layout=widgets.Layout(height=WIDGET_HEIGHT))
        self.max_widget = widgets.HTML(value=self._to_string(slider_max), layout=widgets.Layout(height=WIDGET_HEIGHT))
        self.slider_box = widgets.HBox([self.min_widget, self.slider, self.max_widget, self.reset_button],
                                        layout=widgets.Layout(width="90%", margin="auto"))
        super().__init__(**kwargs)

    @staticmethod
    def _to_string(value):
        """Convert a value to a string."""
        # Unify casting rule for any provided numeric data type
        return f"{value:.8g}"

    def on_slider_change(self, event):
        """Handle slider value on its change."""
        if self.slide_fn is not None:
            self.slide_fn(event)

    def on_reset(self, event):
        """Reset slider to its initial value."""
        if self.reset_fn is not None:
            return self.reset_fn(event)
        return self.set_slider(value=self.slider_init)

    def construct_header(self):
        """Append the slider below the plot header."""
        return widgets.VBox([super().construct_header(), self.slider_box], layout=widgets.Layout(overflow="hidden"))

    def set_slider(self, value=None, min=None, max=None, step=None, **kwargs):
        """Change value, limits, step or other slider state."""
        min = self.slider.min if min is None else min
        max = self.slider.max if max is None else max
        step = self.slider.step if step is None else step
        if value is None:
            current_value = self.slider.value
            value = (min + max) / 2 if min > current_value or max < current_value else current_value

        self.slider.set_state({"min": min, "max": max, "value": value, "step": step, **kwargs})
        self.slider_init = value
        self.min_widget.value = self._to_string(self.slider.min)
        self.max_widget.value = self._to_string(self.slider.max)


class PairedPlot:
    """Construct a plot that contains two interactive plots stacked together.

    Usually one wants to display a clickable plot (`main`) which updates an auxiliary plot (`aux`) on each click. In
    this case both plots may need to have access to the current state of each other. `PairedPlot` can be treated as
    such a state container: if its bound method is used as a `plot_fn`/`click_fn`/`unclick_fn` of `main` or `aux` plots
    it gets access to both `InteractivePlot`s and all the attributes created in `PairedPlot.__init__`.

    Parameters
    ----------
    orientation : {"horizontal", "vertical"}, optional, defaults to "horizontal"
        Defines whether to stack the main and auxiliary plots horizontally or vertically.

    Attributes
    ----------
    main : InteractivePlot
        The main plot.
    aux : InteractivePlot
        The auxiliary plot.
    box : ipywidgets.widgets.widget_box.Box
        A container that stores boxes of both `main` and `aux`.
    """
    def __init__(self, orientation="horizontal"):
        if orientation == "horizontal":
            box_type = widgets.HBox
        elif orientation == "vertical":
            box_type = widgets.VBox
        else:
            raise ValueError("Unknown plot orientation, must be either 'horizontal' or 'vertical'")

        self.main = self.construct_main_plot()
        self.aux = self.construct_aux_plot()
        self.box = box_type([self.main.box, self.aux.box])

    def construct_main_plot(self):
        """Construct the main plot. Must be overridden in child classes."""
        raise NotImplementedError

    def construct_aux_plot(self):
        """Construct the auxiliary plot. Must be overridden in child classes."""
        raise NotImplementedError

    def plot(self):
        """Display the paired plot."""
        self.main.plot(display_box=False)
        self.aux.plot(display_box=False)
        display(self.box)


def attach_widget(widget, widget_to_attach, position, **kwargs):
    """Construct flexible box from two provided widgets. `position` argument defines widgets relation, thus
    `position="top"` for example, results in flexible box where `widget_to_attach` will be placed on top of
    the `widget`."""
    if position == "top":
        return widgets.VBox([widget_to_attach, widget], **kwargs)
    if position == "bottom":
        return widgets.VBox([widget, widget_to_attach], **kwargs)
    if position == "left":
        return widgets.HBox([widget_to_attach, widget], **kwargs)
    return widgets.HBox([widget, widget_to_attach], **kwargs)
