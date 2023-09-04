""""Base class for various transforms of seismic wavefield. """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .utils import add_colorbar, set_ticks, set_text_formatting, to_list
from .decorators import plotter

class Spectrum:
    def __init__(self, spectrum, x_values, y_values):
        self.spectrum = spectrum
        self.x_values = x_values
        self.y_values = y_values

    def normalize(self):
        spec_squares = np.power(self.spectrum, 2, dtype=np.float64)
        spectrum_max = (np.nansum(spec_squares, axis=1, keepdims=True) ** 0.5).astype(np.float32)
        self.spectrum = np.where(spectrum_max != 0, self.spectrum / spectrum_max, 0)
        return self


    @plotter(figsize=(10, 9))
    def plot(self, vfunc=None, align_vfunc=True, grid=False, colorbar=True, x_label=None, x_ticker=None, y_label=None, y_ticker=None,
             title=None, clip_threshold_quantile=0.99, n_levels=10, ax=None, **kwargs):
        # Cast text-related parameters to dicts and add text formatting parameters from kwargs to each of them
        (title, x_ticker, y_ticker), kwargs = set_text_formatting(title, x_ticker, y_ticker, **kwargs)

        cmap = plt.get_cmap('seismic')
        level_values = np.linspace(np.quantile(self.spectrum, 1 - clip_threshold_quantile), np.quantile(self.spectrum, clip_threshold_quantile), n_levels)
        norm = mcolors.BoundaryNorm(level_values, cmap.N, clip=True)
        extent=[self.x_values[0], self.x_values[-1], self.y_values[-1], self.y_values[0]]
        # img = ax.imshow(self.spectrum, norm=norm, aspect='auto', cmap=cmap, extent=extent)
        from matplotlib.image import NonUniformImage
        img = NonUniformImage(ax, cmap=cmap, norm=norm, extent=extent, interpolation='bilinear')
        img.set_data(self.x_values, self.y_values, self.spectrum)
        ax.add_image(img)
        ax.set_xlim(self.x_values[0], self.x_values[-1])
        ax.set_ylim(self.y_values[-1], self.y_values[0])
    
        add_colorbar(ax, img, colorbar, y_ticker=y_ticker)
        ax.set_title(**{"label": None, **title})

        if vfunc is not None:
            for ix_vfunc in to_list(vfunc):
                if align_vfunc:
                    ix_vfunc = ix_vfunc.copy().recalculate(self.y_values[0], self.y_values[-1])
                ix_vfunc.plot(ax=ax, invert=False, plot_bounds=True, linewidth=2.5, marker="o", markevery=slice(1, -1), fill_area_color='white')

        if grid:
            ax.grid(c='k')

        # set_ticks(ax, "x", x_label, self.x_values, **x_ticker)
        # set_ticks(ax, "y", "Time", self.y_values, **y_ticker)