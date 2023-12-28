"""Base class for the representation of seismic gather in different domains. """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.image import NonUniformImage

from .utils import add_colorbar, set_ticks, set_text_formatting, to_list
from .decorators import batch_method, plotter


class Spectrum:
    """Base class for various transforms of seismic wavefield. 
    Implements general processing and visualization logic.

    Parameters
    ----------
    spectrum : 2d np.ndarray
        Spectrum values.
    x_values : 1d np.array
        Unit values for spectrum x axis.
    y_values : 1d np.array
        Unit values for spectrum y axis.
    coords : Coordinates or None, optional, defaults to None
        Spatial coordinates of the spectrum.

    Attributes
    ----------
    spectrum : 2d np.ndarray
        Spectrum values.
    x_values : 1d np.array
        Unit values for spectrum x axis.
    y_values : 1d np.array
        Unit values for spectrum y axis.
    coords : Coordinates or None
        Spatial coordinates of the spectrum.
    """
    def __init__(self, spectrum, x_values, y_values, coords=None):
        self.spectrum = spectrum
        self.x_values = x_values
        self.y_values = y_values
        self.coords = coords


    @property
    def sample_interval(self):
        """ Sample interval of spectrum y_values. None if the axis is not uniform. """
        dy = np.diff(self.y_values)
        if np.allclose(dy, dy[0]):
            return dy[0]
        else: 
            return None


    @property
    def is_y_axis_uniform(self):
        return self.sample_interval is not None


    @property
    def is_x_axis_uniform(self):
        dx = np.diff(self.x_values)
        return np.allclose(dx, dx[0])


    @property
    def are_axes_uniform(self):
        return self.is_x_axis_uniform and self.is_y_axis_uniform


    @batch_method(target="t", copy_src=False)
    def scale_norm(self):
        """ Scale the spectrum along the y axis. """
        l2_norm = np.nansum(self.spectrum ** 2, axis=1, keepdims=True) ** 0.5
        self.spectrum = np.where(spectrum_max != 0, self.spectrum / l2_norm, 0)
        return self


    @plotter(figsize=(10, 9))
    def plot(self, vfunc=None, align_vfunc=True, grid=False, colorbar=True, x_label=None, x_ticker=None, y_label=None, y_ticker=None,
             title=None, clip_threshold_quantile=0.99, n_levels=10, ax=None,  interpolation=None, **kwargs):
        """Plot spectrum and, optionally, vfuncs on on it.

        Parameters
        ----------
        vfunc: VFUNC, iterable of VFUNC, optional, defaults to None
            VFUNCs to be plotted on the spectrum.
        align_vfunc: bool, optional, defaults to True
            Whether aligh (cut or extend) VFUNC y_axis to spectrum y_axis.
        grid : bool, optional, defaults to False
            Specifies whether to draw a grid on the plot.
        colorbar : bool or dict, optional, defaults to True
            Whether to add a colorbar to the right of the velocity spectrum plot.
            If `dict`, defines extra keyword arguments for `matplotlib.figure.Figure.colorbar`.
        x_label : str, optional, defaults to None
            The title of the x-axis.
        x_ticklabels : list of str, optional, defaults to None
            An array of labels for the x-axis.
        x_ticker : dict, optional, defaults to None
            Parameters for ticks and ticklabels formatting for the x-axis; see `.utils.set_ticks` for more details.
        y_ticklabels : list of str, optional, defaults to None
            An array of labels for the y-axis.            
        title : str, optional, defaults to None
            Plot title.
        clip_threshold_quantile : float, optional, defaults to 0.99
            Clip the velocity spectrum values by given quantile.
        n_levels : int, optional, defaults to 10
            The number of levels on the colorbar.
        ax : matplotlib.axes.Axes, optional, defaults to None
            Axes of the figure to plot on.
        interpolation: str, optional, defaults to None
            Interpolation method either `ax.imshow` for case uniform spectrum
            or `NonUniformImage` in case non-uniform spectrum.
        kwargs : misc, optional
            Additional common keyword arguments for `x_ticker` and `y_tickers`.
        """

        # Cast text-related parameters to dicts and add text formatting parameters from kwargs to each of them
        (title, x_ticker, y_ticker), kwargs = set_text_formatting(title, x_ticker, y_ticker, **kwargs)

        cmap = plt.get_cmap('seismic')
        level_values = np.linspace(np.quantile(self.spectrum, 1 - clip_threshold_quantile), np.quantile(self.spectrum, clip_threshold_quantile), n_levels)
        norm = mcolors.BoundaryNorm(level_values, cmap.N, clip=True)
        extent=[self.x_values[0], self.x_values[-1], self.y_values[-1], self.y_values[0]]

        if self.are_axes_uniform:
            img = ax.imshow(self.spectrum, norm=norm, cmap=cmap, extent=extent, aspect='auto', interpolation=interpolation)
        else:
            img = NonUniformImage(ax, norm=norm, cmap=cmap, extent=extent, interpolation=interpolation)
            img.set_data(self.x_values, self.y_values, self.spectrum)
            ax.add_image(img)
        
        ax.set_xlim(self.x_values[0], self.x_values[-1])
        ax.set_ylim(self.y_values[-1], self.y_values[0])
    
        add_colorbar(ax, img, colorbar, y_ticker=y_ticker)
        ax.set_title(**{"label": None, **title})

        if vfunc is not None:
            for ix_vfunc in to_list(vfunc):
                if align_vfunc:
                    ix_vfunc = ix_vfunc.copy().crop(self.y_values[0], self.y_values[-1])
                ix_vfunc.plot(ax=ax, invert=False, plot_bounds=True, linewidth=2.5, marker="o", markevery=slice(1, -1), fill_area_color='white')

        if grid:
            ax.grid(c='k')

        set_ticks(ax, "x", x_label, self.x_values, axes_has_units=True, **x_ticker)
        set_ticks(ax, "y", "Time", self.y_values, axes_has_units=True, **y_ticker)