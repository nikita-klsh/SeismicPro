"""Implements interactive plots of vertical velocity spectrum and residual velocity spectrum."""

from functools import partial

import numpy as np

from ..utils import get_text_formatting_kwargs
from ..utils.interactive_plot_utils import InteractivePlot, PairedPlot


class VelocitySpectrumPlot(PairedPlot):  # pylint: disable=too-many-instance-attributes
    """Define an interactive velocity spectrum plot.

    This plot also displays the gather used to calculate the velocity spectrum. Clicking on velocity spectrum highlight
    the corresponding hodograph on the gather plot and allows performing NMO or LMO correction of the gather with
    the selected velocity by switching the view. The width of the hodograph matches the window size used to calculate
    the spectrum on both views. An initial click is performed on the maximum spectrum value.
    """
    def __init__(self, velocity_spectrum, half_win_size=10, title=None, gather_plot_kwargs=None,
                 figsize=(4.5, 4.5), fontsize=8, orientation="horizontal", **kwargs):
        kwargs = {"fontsize": fontsize, **kwargs}
        text_kwargs = get_text_formatting_kwargs(**kwargs)
        if gather_plot_kwargs is None:
            gather_plot_kwargs = {}
        self.gather_plot_kwargs = {"title": None, **text_kwargs, **gather_plot_kwargs}

        self.figsize = figsize
        self.orientation = orientation
        self.title = title
        self.click_time = None
        self.click_vel = None
        self.velocity_spectrum = velocity_spectrum
        self.gather = self.velocity_spectrum.gather.copy(ignore="data").sort('offset')
        self.plot_velocity_spectrum = partial(velocity_spectrum.plot, title="", **kwargs)
        self.half_win_size = half_win_size

        super().__init__(orientation=orientation)

    def construct_main_plot(self):
        """Construct a clickable velocity spectrum plot."""
        return InteractivePlot(plot_fn=self.plot_velocity_spectrum, click_fn=self.click, unclick_fn=self.unclick,
                               title=self.title, figsize=self.figsize)

    def construct_aux_plot(self):
        """Construct a correctable gather plot."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        plotter = InteractivePlot(plot_fn=[self.plot_gather, partial(self.plot_gather, corrected=True)],
                                  title=self.get_gather_title, figsize=self.figsize, toolbar_position=toolbar_position)
        plotter.view_button.disabled = True
        return plotter

    def get_gather_title(self):
        """Get title of the gather plot."""
        if (self.click_time is None) or (self.click_vel is None):
            return "Gather"
        return f"Hodograph from {self.click_time:.0f} ms with {self.click_vel:.2f} m/s velocity"

    def get_gather(self, corrected=False):
        """Get an optionally corrected gather."""
        raise NotImplementedError

    @staticmethod
    def hodograph_func(t0, x, v):
        """Compute hodograph times for given t0, offsets and velocity."""
        raise NotImplementedError

    def get_hodograph_times(self, corrected):
        """Get hodograph times if a click has been performed."""
        if (self.click_time is None) or (self.click_vel is None):
            return None
        if not corrected:
            return self.hodograph_func(self.click_time, self.gather.offsets, self.click_vel / 1000)
        return np.full_like(self.gather.offsets, self.click_time)

    def plot_gather(self, ax, corrected=False):
        """Plot the gather and a hodograph if click has been performed."""
        gather = self.get_gather(corrected=corrected)
        gather.plot(ax=ax, **self.gather_plot_kwargs)

        hodograph_times = self.get_hodograph_times(corrected=corrected)
        if hodograph_times is None:
            return
        self.plot_hodograph(ax, hodograph_times)

    def plot_hodograph(self, ax, hodograph_times, color="tab:blue", mask=None, label=None):
        """Highlight the hodograph on the gather."""
        hodograph_low = np.clip(self.gather.times_to_indices(hodograph_times - self.half_win_size) - 0.5,
                                0, self.gather.n_times - 1)
        hodograph_high = np.clip(self.gather.times_to_indices(hodograph_times + self.half_win_size) - 0.5,
                                0, self.gather.n_times - 1)
        ax.fill_between(np.arange(len(hodograph_times)), hodograph_low, hodograph_high,
                        mask, color=color, alpha=0.5, label=label)

    def get_velocity_time_by_coords(self, coords):
        """ Transform click coords to units."""
        return coords[0], coords[1]

    def click(self, coords):
        """Highlight the hodograph defined by click location on the gather plot."""
        self.aux.view_button.disabled = False
        self.click_vel, self.click_time = self.get_velocity_time_by_coords(coords)
        self.aux.redraw()
        return coords

    def unclick(self):
        """Remove the highlighted hodograph and switch to a non-corrected view."""
        self.click_time = None
        self.click_vel = None
        self.aux.set_view(0)
        self.aux.view_button.disabled = True


class VerticalVelocitySpectrumPlot(VelocitySpectrumPlot):
    """Interactive Vertical Velocity Spectrum plot."""

    def get_gather(self, corrected=False):
        """Get an optionally corrected gather."""
        if not corrected:
            return self.gather
        max_stretch_factor = self.velocity_spectrum.max_stretch_factor
        return self.gather.copy(ignore=["headers", "data", "samples"]) \
                          .apply_nmo(self.click_vel, max_stretch_factor=max_stretch_factor)

    @staticmethod
    def hodograph_func(t0, x, v):
        """Hyperbolic hodograph times computation."""
        return (t0 ** 2 + (x/v) ** 2) ** 0.5

    def plot_hodograph(self, ax, hodograph_times):
        """Plot hodograph and highlight it's stretch and non-stretch zones."""
        max_offset = self.click_time * self.click_vel * \
                     np.sqrt((1 + self.velocity_spectrum.max_stretch_factor)**2 - 1) / 1000
        super().plot_hodograph(ax, hodograph_times, "tab:blue", self.gather.offsets < max_offset, 'non-stretch muted')
        if not np.isinf(self.velocity_spectrum.max_stretch_factor):
            super().plot_hodograph(ax, hodograph_times, "tab:red", self.gather.offsets > max_offset, 'stretch muted')
            ax.legend(loc='upper right', fontsize='small')


class SlantStackPlot(VelocitySpectrumPlot):
    """Interactive Slant Stack plot."""

    def get_gather(self, corrected=False):
        """Get an optionally corrected gather."""
        if not corrected:
            return self.gather
        return self.gather.copy(ignore=["headers", "data", "samples"]) \
                          .apply_lmo(self.click_vel, 0)

    @staticmethod
    def hodograph_func(t0, x, v):
        """Linear hodograph times computation."""
        return t0 + x/v


class RedidualVelocitySpectrumPlot(VerticalVelocitySpectrumPlot):
    """Interactive Residual Velocity Spectrum plot."""

    def get_velocity_time_by_coords(self, coords):
        """Cast (margin, time) to (velocity, time)."""
        click_margin, click_time = coords[0], coords[1]
        click_vel = self.velocity_spectrum.stacking_velocity(click_time) * (1 + click_margin)
        return click_vel, click_time
