from functools import partial

import numpy as np

from ..velocity_spectrum.interactive_plot import VelocitySpectrumPlot
from ..utils.interactive_plot_utils import InteractivePlot
from ..utils import add_colorbar


class DispersionSpectrumPlot(VelocitySpectrumPlot):
    
    def construct_aux_plot(self):
        """Construct a correctable gather plot."""
        toolbar_position = "right" if self.orientation == "horizontal" else "left"
        plotter = InteractivePlot(plot_fn=[self.plot_gather, partial(self.plot_gather, corrected=True), self.plot_phases, partial(self.plot_phases, unwrap=True), self.cumsum, self.cumsum_abs],
                                  title=self.get_gather_title, figsize=self.figsize, toolbar_position=toolbar_position)
        plotter.view_button.disabled = True
        return plotter

    def get_gather_title(self):
        """Get title of the gather plot."""
        if (self.click_time is None) or (self.click_vel is None):
            return "Gather"
        fft, f = self.velocity_spectrum.calculate_ft(self.gather.data, self.gather.sample_interval/1000, self.velocity_spectrum.frequencies[-1])
        i = np.argmin(abs(f - self.click_time))
        self.click_time = f[i]
        return f"Hodographs with freq {self.click_time:.2f} HZ with {self.click_vel:.2f} km/s velocity"

    def plot_gather(self, ax, corrected=False):
        """Plot the gather and a hodograph if click has been performed."""
        if not corrected:
            self.gather.plot(ax=ax)
            return
        if (self.click_time is None) or (self.click_vel is None):
            self.gather.plot(ax=ax)
            return
        gather = self.gather.copy()
        # dt = 400
        # hodographs = [np.array(t0 + self.gather.offsets/self.click_vel) for t0 in np.arange(-self.gather.times[-1], self.gather.times[-1], dt)]
        # for hodograph in hodographs:
        #     hodograph_y = self.gather.times_to_indices(hodograph) - 0.5  # Correction for pixel center
        #     half_window = dt / 2 / 2 / self.gather.sample_interval
        #     hodograph_low = np.clip(hodograph_y - half_window, 0, len(self.gather.times) - 1)
        #     hodograph_high = np.clip(hodograph_y + half_window, 0, len(self.gather.times) - 1)
        #     ax.fill_between(np.arange(len(hodograph)), hodograph_low, hodograph_high, color="tab:blue", alpha=0.3)

        gather.bandpass_filter(low=self.click_time - 1, high=self.click_time + 1, filter_size=81 * 2)
        gather.apply_lmo(refractor_velocity=self.click_vel, delay=0, fill_value=np.nan)
        gather.plot(ax=ax, **self.gather_plot_kwargs) # gather_plot_kwargs # q_vmin=0.01, q_vmax=0.99

    def plot_phases(self, ax, unwrap=False):
        if (self.click_time is None) or (self.click_vel is None):
            return
        gather = self.gather
        fft, f = self.velocity_spectrum.calculate_ft(gather.data, gather.sample_interval/1000, self.velocity_spectrum.frequencies[-1])
        freq, v = self.click_time, self.click_vel
        
        i = np.argmin(abs(f - freq))
        f = f[i]

        phases = fft[:, i]
        phases = phases / np.abs(phases)

        phi = np.pi * 0.
        exp = 1j * (phi + 2 * np.pi * f * self.gather.offsets / (v))

        est = np.exp(-exp)
        shifts = phases * np.exp(exp) 

        phases = np.angle(phases)
        if unwrap:
            phases = np.unwrap(phases)
        ax.scatter(gather.offsets, phases, label='CDP', s=15)

        est = np.angle(est)
        if unwrap:
            est = np.unwrap(est)
        ax.scatter(gather.offsets, est, label='EST', s=15)

        metric = np.abs(np.mean(shifts))
        ax.set_title(f'{metric:0.2f}')

        # shifts = np.angle(shifts) 
        # ax.scatter(gather.offsets, shifts, label='SHIFT', s=10, c='k')
        if self.velocity_spectrum.start is not None and self.velocity_spectrum.end is not None:
            ax.axvline(self.velocity_spectrum.start(f)), ax.axvline(self.velocity_spectrum.end(f))
        ax.legend()

    def cumsum(self, ax):
        import matplotlib.colors as mcolors
        from matplotlib.patches import Circle
        colors = ((0.0, 0.6, 0.0), (.66, 1, 0), (0.9, 0.0, 0.0))
        cmap = mcolors.LinearSegmentedColormap.from_list("cmap", colors, N=5)
        gather = self.gather
        fft, f = self.velocity_spectrum.calculate_ft(gather.data, gather.sample_interval/1000, self.velocity_spectrum.frequencies[-1])
        freq, v = self.click_time, self.click_vel
        
        i = np.argmin(abs(f - freq))
        f = f[i]

        phases = fft[:, i]
        phases = phases / np.abs(phases)

        phi = np.pi * 0.
        exp = 1j * (phi + 2 * np.pi * f * self.gather.offsets / (v))

        est = np.exp(-exp)
        shifts = phases * np.exp(exp)
        
        if self.velocity_spectrum.start is not None and self.velocity_spectrum.end is not None:
            start, end = self.velocity_spectrum.start(f), self.velocity_spectrum.end(f)
            mask = (self.gather.offsets >= start) & (self.gather.offsets <= end)
            s = np.where(mask, 15, 5)
        else:
            s = 10

        im = ax.scatter(np.cumsum(shifts).real , np.cumsum(shifts).imag , s=s, c=self.gather.offsets, cmap=cmap)
        ax.axis('equal'), ax.grid(), ax.scatter(0, 0, s=50, marker='o', c='b')
        add_colorbar(ax, im, True)


    def cumsum_abs(self, ax):
        import matplotlib.colors as mcolors
        from matplotlib.patches import Circle
        colors = ((0.0, 0.6, 0.0), (.66, 1, 0), (0.9, 0.0, 0.0))
        cmap = mcolors.LinearSegmentedColormap.from_list("cmap", colors, N=5)
        gather = self.gather
        fft, f = self.velocity_spectrum.calculate_ft(gather.data, gather.sample_interval/1000, self.velocity_spectrum.frequencies[-1])
        freq, v = self.click_time, self.click_vel
        
        i = np.argmin(abs(f - freq))
        f = f[i]

        phases = fft[:, i]
        phases = phases / np.abs(phases)

        phi = np.pi * 0.
        exp = 1j * (phi + 2 * np.pi * f * self.gather.offsets / (v))

        est = np.exp(-exp)
        shifts = phases * np.exp(exp) 
        im = ax.scatter(self.gather.offsets, np.abs(np.cumsum(shifts)), s=10, c=self.gather.offsets, cmap=cmap)
        if self.velocity_spectrum.start is not None and self.velocity_spectrum.end is not None:
            ax.axvline(self.velocity_spectrum.start(f)), ax.axvline(self.velocity_spectrum.end(f))
        add_colorbar(ax, im, True)
