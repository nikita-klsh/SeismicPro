""""Base class for various transforms of seismic wavefield. """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .utils import add_colorbar, set_ticks, set_text_formatting

class Spectrum:
    def __init__(self, spectrum, x_values, y_values):
        self.spectrum = spectrum
        self.x_values = x_values
        self.y_values = y_values

    
    def normalize(self):
        spectrum_max = np.nansum(self.spectrum ** 2, axis=1, keepdims=True) ** 0.5
        self.spectrum = np.where(spectrum_max != 0, self.spectrum / spectrum_max, 0)
        return self

    def plot(self, title=None, x_label=None, x_ticklabels=None, x_ticker=None, y_ticklabels=None, y_ticker=None,
              grid=False, stacking_velocity_ix=None, velocity_bounds_ix=None, colorbar=True,
              clip_threshold_quantile=0.99, n_levels=10, ax=None, **kwargs):
        # Cast text-related parameters to dicts and add text formatting parameters from kwargs to each of them
        (title, x_ticker, y_ticker), kwargs = set_text_formatting(title, x_ticker, y_ticker, **kwargs)

        cmap = plt.get_cmap('seismic')
        level_values = np.linspace(np.quantile(self.spectrum, 1 - clip_threshold_quantile), np.quantile(self.spectrum, clip_threshold_quantile), n_levels)
        norm = mcolors.BoundaryNorm(level_values, cmap.N, clip=True)
        img = ax.imshow(self.spectrum, norm=norm, aspect='auto', cmap=cmap)
        add_colorbar(ax, img, colorbar, y_ticker=y_ticker)
        ax.set_title(**{"label": None, **title})

        if stacking_velocity_ix is not None:
            stacking_times_ix, stacking_velocities_ix = stacking_velocity_ix
            ax.plot(stacking_velocities_ix, stacking_times_ix, c='#fafcc2', linewidth=2.5,
                    marker="o", markevery=slice(1, -1))
        if velocity_bounds_ix is not None:
            ax.fill_betweenx(*velocity_bounds_ix, color="white", alpha=0.2)
        if grid:
            ax.grid(c='k')

        
        set_ticks(ax, "x", x_label, self.x_values, **x_ticker)
        set_ticks(ax, "y", "Time", self.y_values, **y_ticker)