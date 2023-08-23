"""Implements VerticalVelocitySpectrum and ResidualVelocitySpectrum classes."""

# pylint: disable=not-an-iterable, too-many-arguments
import math
from functools import partial

import numpy as np
from numba import njit, prange
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

from .utils import coherency_funcs
from .interactive_plot import VerticalVelocitySpectrumPlot, RedidualVelocitySpectrumPlot, SlantStackPlot
from ..spectrum import Spectrum
from ..containers import SamplesContainer
from ..decorators import batch_method, plotter
from ..stacking_velocity import StackingVelocity, StackingVelocityField
from ..utils import add_colorbar, set_ticks, set_text_formatting, get_first_defined, VFUNC
from ..gather.utils.correction import apply_constant_velocity_nmo, apply_constant_velocity_lmo
from ..const import DEFAULT_STACKING_VELOCITY


COHERENCY_FUNCS = {
    "stacked_amplitude": coherency_funcs.stacked_amplitude,
    "S": coherency_funcs.stacked_amplitude,
    "normalized_stacked_amplitude": coherency_funcs.normalized_stacked_amplitude,
    "NS": coherency_funcs.normalized_stacked_amplitude,
    "semblance": coherency_funcs.semblance,
    "NE": coherency_funcs.semblance,
    'crosscorrelation': coherency_funcs.crosscorrelation,
    'CC': coherency_funcs.crosscorrelation,
    'ENCC': coherency_funcs.energy_normalized_crosscorrelation,
    'energy_normalized_crosscorrelation': coherency_funcs.energy_normalized_crosscorrelation
}


class BaseVelocitySpectrum(Spectrum):

    @property
    def velocities(self):
        return self.x_values

    @property
    def times(self):
        return self.y_values

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def calc_single_velocity_spectrum(coherency_func, gather_data, times, offsets, velocity, sample_interval, delay,
                                      half_win_size_samples, t_min_ix, t_max_ix,
                                      interpolate=True, correction_func=None, correction_func_args=None, out=None):
        t_win_size_min_ix = max(0, t_min_ix - half_win_size_samples)
        t_win_size_max_ix = min(len(times) - 1, t_max_ix + half_win_size_samples)

        corrected_gather_data = correction_func(gather_data, offsets, sample_interval, delay,
                                                times[t_win_size_min_ix: t_win_size_max_ix + 1], velocity,
                                                interpolate, np.nan, *correction_func_args)

        numerator, denominator = coherency_func(corrected_gather_data)

        if out is None:
            out = np.empty(t_max_ix - t_min_ix, dtype=np.float32)

        for t in prange(t_min_ix, t_max_ix):
            t_rel = t - t_win_size_min_ix
            ix_from = max(0, t_rel - half_win_size_samples)
            ix_to = min(corrected_gather_data.shape[1] - 1, t_rel + half_win_size_samples + 1)
            out[t - t_min_ix] = np.sum(numerator[ix_from : ix_to]) / (np.sum(denominator[ix_from : ix_to]) + 1e-8)
        return out

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def _calc_spectrum_numba(spectrum_func, coherency_func, gather_data, times, offsets, velocities, sample_interval,
                             delay, half_win_size_samples, interpolate, correction_func, correction_func_args):
        velocity_spectrum = np.empty((gather_data.shape[1], len(velocities)), dtype=np.float32)
        for j in prange(len(velocities)):  # pylint: disable=consider-using-enumerate
            spectrum_func(coherency_func=coherency_func, gather_data=gather_data, times=times, offsets=offsets,
                          velocity=velocities[j], sample_interval=sample_interval, delay=delay,
                          half_win_size_samples=half_win_size_samples, t_min_ix=0, t_max_ix=gather_data.shape[1],
                          interpolate=interpolate, out=velocity_spectrum[:, j],
                          correction_func=correction_func, correction_func_args=correction_func_args)
        return velocity_spectrum


class VerticalVelocitySpectrum(BaseVelocitySpectrum):

    @property
    def n_velocities(self):
        """int: The number of velocities the spectrum was calculated for."""
        return len(self.velocities)

    
    @classmethod
    def from_gather(cls, gather, velocities=None, stacking_velocity=None, relative_margin=0.2, velocity_step=50,
                    window_size=50, mode='semblance', max_stretch_factor=np.inf, interpolate=True):
            
        half_win_size_samples = math.ceil((window_size / gather.sample_interval / 2))

        coherency_func = COHERENCY_FUNCS.get(mode)
        if coherency_func is None:
            raise ValueError(f"Unknown mode {mode}, available modes are {COHERENCY_FUNCS.keys()}")
    
        if stacking_velocity is None:
            stacking_velocity = DEFAULT_STACKING_VELOCITY
        if isinstance(stacking_velocity, StackingVelocityField):
            stacking_velocity = stacking_velocity(gather.coords)
        if velocities is None:
            velocities = cls.get_velocity_range(gather.times, stacking_velocity, relative_margin, velocity_step)
        else:
            velocities = np.sort(velocities)

        velocities = np.asarray(velocities, dtype=np.float32)  # m/s


        velocities_ms = velocities / 1000  # from m/s to m/ms
        kwargs = {"spectrum_func": cls.calc_single_velocity_spectrum, "coherency_func": coherency_func,
                  "gather_data": gather.data, "times": gather.times, "offsets": gather.offsets,
                  "velocities": velocities_ms, "sample_interval": gather.sample_interval, "delay": gather.delay,
                  "half_win_size_samples": half_win_size_samples,
                  "interpolate": interpolate, 
                  "correction_func": apply_constant_velocity_nmo, "correction_func_args": (max_stretch_factor, )}
        
        velocity_spectrum = cls._calc_spectrum_numba(**kwargs)
        spectrum = cls(velocity_spectrum, velocities, gather.times)
        spectrum.gather = gather.copy()
        spectrum.times_interval = gather.sample_interval
        spectrum.coords = gather.coords
    
        spectrum.stacking_velocity = stacking_velocity
        spectrum.relative_margin = relative_margin
    
        spectrum.coherency_func = coherency_func
        spectrum.half_win_size_samples = half_win_size_samples
        spectrum.correction_func_args = (max_stretch_factor, )
        return spectrum


    @staticmethod
    def get_velocity_range(times, stacking_velocity, relative_margin, velocity_step):
        """Return an array of stacking velocities for spectrum calculation:
        1. First `stacking_velocity` is evaluated for gather times to estimate the velocity range being examined.
        2. Then the range is additionally extended by `relative_margin` * 100% in both directions.
        3. The resulting velocities are then evenly sampled from this range with a step of `velocity_step`.
        """
        interpolated_velocities = stacking_velocity(times)
        min_velocity = np.min(interpolated_velocities) * (1 - relative_margin)
        max_velocity = np.max(interpolated_velocities) * (1 + relative_margin)
        n_velocities = math.ceil((max_velocity - min_velocity) / velocity_step) + 1
        return min_velocity + velocity_step * np.arange(n_velocities)


    @plotter(figsize=(10, 9), args_to_unpack="stacking_velocity")
    def plot(self, stacking_velocity=None, *, interactive=False, plot_bounds=True, title=None, grid=False, colorbar=True, 
             x_ticker=None, y_ticker=None, **kwargs):
        if title is None:
            title = f"Vertical Velocity Spectrum \n Coherency func: {self.coherency_func.__name__}"
        if isinstance(stacking_velocity, StackingVelocityField):
            stacking_velocity = stacking_velocity(self.coords)
        plot_kwargs = {"vfunc": stacking_velocity, "plot_bounds": plot_bounds, "title": title, 
                      "x_label": "Velocity, m/s", "y_label": 'Time, ms', "grid": grid, "colorbar": colorbar,
                      "x_ticker": x_ticker, "y_ticker": y_ticker, **kwargs} 

        if not interactive:
            return super().plot(**plot_kwargs)
        return VerticalVelocitySpectrumPlot(self, **plot_kwargs).plot()
    

    @batch_method(target="for", args_to_unpack="init", copy_src=False)
    def calculate_stacking_velocity(self, init=None, bounds=None, relative_margin=None, acceleration_bounds="auto",
                                    times_step=100, max_offset=5000, hodograph_correction_step=25, max_n_skips=2):
        kwargs = {"init": get_first_defined(init, self.stacking_velocity), "bounds": bounds,
                  "relative_margin": get_first_defined(relative_margin, self.relative_margin),
                  "acceleration_bounds": acceleration_bounds, "times_step": times_step, "max_offset": max_offset,
                  "hodograph_correction_step": hodograph_correction_step, "max_n_skips": max_n_skips}
        return StackingVelocity.from_vertical_velocity_spectrum(self, **kwargs)


class ResidualVelocitySpectrum(BaseVelocitySpectrum):

    @property
    def margins(self):
        return self.y_values

    @property
    def n_margins(self):
        """int: The number of velocity margins the spectrum was calculated for."""
        return len(self.margins)
    
    
    @classmethod
    def from_gather(cls, gather, stacking_velocity, relative_margin=0.2, velocity_step=25, window_size=50,
                 mode='semblance', max_stretch_factor=np.inf, interpolate=True):
        half_win_size_samples = math.ceil((window_size / gather.sample_interval / 2))

        coherency_func = COHERENCY_FUNCS.get(mode)
        if coherency_func is None:
            raise ValueError(f"Unknown mode {mode}, available modes are {COHERENCY_FUNCS.keys()}")

        if isinstance(stacking_velocity, StackingVelocityField):
            stacking_velocity = stacking_velocity(gather.coords)

        stacking_velocities = stacking_velocity(gather.times)
        kwargs = {"spectrum_func": cls.calc_single_velocity_spectrum, "coherency_func": coherency_func,
                  "gather_data": gather.data, "times": gather.times, "offsets": gather.offsets,
                  "stacking_velocities": stacking_velocities, "relative_margin": relative_margin,
                  "velocity_step": velocity_step, "sample_interval": gather.sample_interval, "delay": gather.delay,
                  "half_win_size_samples": half_win_size_samples,
                  "interpolate": interpolate, 
                  "correction_func": apply_constant_velocity_nmo, "correction_func_args": (max_stretch_factor, )}

        
        velocity_spectrum = cls._calc_spectrum_numba(**kwargs)
        margins, margin_step = np.linspace(-relative_margin, relative_margin, velocity_spectrum.shape[1], retstep=True)
        spectrum = cls(velocity_spectrum, margins, gather.times)
        spectrum.gather = gather.copy()
        
        spectrum.stacking_velocity = stacking_velocity.copy().recalculate(gather.times[0], gather.times[-1])
        spectrum.coherency_func = coherency_func
        spectrum.half_win_size_samples = half_win_size_samples
        spectrum.correction_func_args = (max_stretch_factor, )
        return spectrum


    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def _calc_spectrum_numba(spectrum_func, coherency_func, gather_data, times, offsets, stacking_velocities,
                             relative_margin, velocity_step, sample_interval, delay, half_win_size_samples,
                             interpolate, correction_func, correction_func_args):

        # Calculate velocity bounds and a range of velocities for residual spectrum calculation
        left_bound = stacking_velocities * (1 - relative_margin)
        right_bound = stacking_velocities * (1 + relative_margin)
        min_velocity = left_bound.min()
        max_velocity = right_bound.max()
        n_velocities = math.ceil((max_velocity - min_velocity) / velocity_step) + 1
        velocities = (min_velocity + velocity_step * np.arange(n_velocities)).astype(np.float32)

        # Convert bounds to their indices in the array of velocities and construct a binary mask that stores True
        # values for (time, velocity) pairs for which spectrum should be calculated
        left_bound_ix = np.empty(len(left_bound), dtype=np.int32)
        right_bound_ix = np.empty(len(right_bound), dtype=np.int32)
        spectrum_mask = np.zeros((gather_data.shape[1], len(velocities)), dtype=np.bool_)
        for i in prange(len(left_bound_ix)):
            left_bound_ix[i] = np.argmin(np.abs(left_bound[i] - velocities))
            right_bound_ix[i] = np.argmin(np.abs(right_bound[i] - velocities))
            spectrum_mask[i, left_bound_ix[i] : right_bound_ix[i] + 1] = True

        # Calculate only necessary part of the vertical velocity spectrum
        velocity_spectrum = np.zeros((gather_data.shape[1], len(velocities)), dtype=np.float32)
        for i in prange(len(velocities)):
            t_ix = np.where(spectrum_mask[:, i])[0]
            if len(t_ix) == 0:
                continue
            t_min_ix = t_ix[0]
            t_max_ix = t_ix[-1]

            spectrum_func(coherency_func=coherency_func, gather_data=gather_data, times=times, offsets=offsets,
                          velocity=velocities[i] / 1000, sample_interval=sample_interval, delay=delay,
                          half_win_size_samples=half_win_size_samples, t_min_ix=t_min_ix, t_max_ix=t_max_ix+1,
                          interpolate=interpolate,
                          out=velocity_spectrum[t_min_ix : t_max_ix + 1, i],
                          correction_func=correction_func, correction_func_args=correction_func_args)

        # Interpolate velocity spectrum to get a rectangular image
        residual_velocity_spectrum_len = (right_bound_ix - left_bound_ix).max()
        residual_velocity_spectrum = np.empty((len(times), residual_velocity_spectrum_len), dtype=np.float32)
        for i in prange(len(residual_velocity_spectrum)):
            cropped_spectrum = velocity_spectrum[i, left_bound_ix[i] : right_bound_ix[i] + 1]
            cropped_velocities = velocities[left_bound_ix[i] : right_bound_ix[i] + 1]
            target_velocities = np.linspace(left_bound[i], right_bound[i], residual_velocity_spectrum_len)
            residual_velocity_spectrum[i] = np.interp(target_velocities, cropped_velocities, cropped_spectrum)
        return residual_velocity_spectrum


    @plotter(figsize=(10, 9))
    def plot(self, *, acceptable_margin=None, title=None, interactive=False, colorbar=True, grid=False,
             x_ticker=None, y_ticker=None, **kwargs):
        if title is None:
            title = f"Residual Velocity Spectrum \n Coherency func: {self.coherency_func.__name__}"
        stacking_velocity = VFUNC(self.stacking_velocity.times, [0] * len(self.stacking_velocity.times))

        if acceptable_margin is not None:
            stacking_velocity.bounds = [VFUNC([0, self.times[-1]], [-acceptable_margin, -acceptable_margin]),
                                        VFUNC([0, self.times[-1]], [acceptable_margin, acceptable_margin])]

        plot_kwargs = {"vfunc": stacking_velocity, "title": title, 
                       "x_label": "Margin, %", "y_label": 'Time, ms', "grid": grid, "colorbar": colorbar,
                       "x_ticker": x_ticker, "y_ticker": y_ticker, **kwargs} 

        if not interactive:
            return super().plot(**plot_kwargs)
        return RedidualVelocitySpectrumPlot(self, **plot_kwargs).plot()


class SlantStack(BaseVelocitySpectrum):
    
    @classmethod
    def from_gather(cls, gather, velocities):
        velocities = np.sort(velocities)
        velocities = np.asarray(velocities, dtype=np.float32)  # m/s
        velocities_ms = velocities / 1000  # from m/s to m/ms

        kwargs = {"spectrum_func": cls.calc_single_velocity_spectrum, "coherency_func": coherency_funcs.stacked_amplitude_sum,
                  "gather_data": gather.data, "times": gather.times, "offsets": gather.offsets,
                  "velocities": velocities_ms, "sample_interval": gather.sample_interval, "delay": gather.delay,
                  "half_win_size_samples": 0,
                  "interpolate": True, 
                  "correction_func": apply_constant_velocity_lmo, "correction_func_args": ()}

        velocity_spectrum = cls._calc_spectrum_numba(**kwargs)
        spectrum =  cls(velocity_spectrum, velocities, gather.times)
        spectrum.gather = gather.copy()
        return spectrum
    

    @plotter(figsize=(10, 9))
    def plot(self, interactive=False, title=None, half_win_size=10, grid=False, colorbar=True, x_ticker=None, y_ticker=None, **kwargs):
        if title is None:
            title = "Slant Stack"

        plot_kwargs = {"title": title, 
                      "x_label": "Velocity, m/s", "y_label": 'Time, ms', "grid": grid, "colorbar": colorbar, 
                      "x_ticker": x_ticker, "y_ticker": y_ticker, **kwargs} 

        if not interactive:
            return super().plot(**plot_kwargs)
        return SlantStackPlot(self, half_win_size=half_win_size, **plot_kwargs).plot()
