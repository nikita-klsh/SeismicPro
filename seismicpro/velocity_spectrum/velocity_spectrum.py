"""Implements VerticalVelocitySpectrum, ResidualVelocitySpectrum and SlantStack classes."""

# pylint: disable=not-an-iterable, too-many-arguments
import math

import numpy as np
from numba import njit, prange

from .utils import coherency_funcs
from .interactive_plot import VerticalVelocitySpectrumPlot, RedidualVelocitySpectrumPlot, SlantStackPlot
from ..spectrum import Spectrum
from ..containers import SamplesContainer
from ..decorators import batch_method, plotter
from ..stacking_velocity import StackingVelocity, StackingVelocityField
from ..utils import get_first_defined, VFUNC
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


class BaseVelocitySpectrum(Spectrum, SamplesContainer):
    """Base class for velocity spectrum calculation.
    Implements general computation logic based on the provided seismic gather.
    """
    def __init__(self, *args, gather=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gather = gather

    @property
    def samples(self):
        """np.ndarray: Array of spectrum's timestamps. Measured in milliseconds."""
        return self.y_values

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def calc_single_velocity_spectrum(coherency_func, correction_type, gather_data, times, offsets, velocity,
                                      sample_interval, delay, half_win_size_samples, t_min_ix, t_max_ix,
                                      interpolate=True, max_strecth_factor=np.inf, out=None):
        """Calculate velocity spectrum for a given range of zero-offset traveltimes and constant velocity.

        Parameters
        ----------
        coherency_func: njitted callable
            A function that estimates hodograph coherency.
        correction_type: str, 'NMO' or 'LMO'
            Type of correction to perform for given velocity. 'NMO' for normal moveout, 'LMO' for linear moveout.
        gather_data : 2d np.ndarray
            Gather data for velocity spectrum calculation.
        times : 1d np.ndarray
            Recording time for each trace value. Measured in milliseconds.
        offsets : array-like
            The distance between source and receiver for each trace. Measured in meters.
        velocity : array-like
            Seismic wave velocity for velocity spectrum computation. Measured in meters/milliseconds.
        sample_interval : float
            Sample interval of seismic traces. Measured in milliseconds.
        delay : float
            Delay recording time of seismic traces. Measured in milliseconds.
        half_win_size_samples : int
            Half of the temporal size for smoothing the velocity spectrum. Measured in samples.
        t_min_ix : int
            Time index in `times` array to start calculating velocity spectrum from. Measured in samples.
        t_max_ix : int
            Time index in `times` array to stop calculating velocity spectrum at. Measured in samples.
        interpolate: bool, optional, defaults to True
            Whether to perform linear interpolation to retrieve amplitudes along hodographs. If `False`, an amplitude
            at the nearest time sample is used.
        max_stretch_factor : float, defaults to np.inf
            Maximum allowable factor for the muter that attenuates the effect of waveform stretching after NMO
            correction. The lower the value, the stronger the mute. In case np.inf (default) no mute is applied.
            Reasonably good value is 0.65. Does not have power in case LMO correction.
        out : np.array, optional
            The buffer to store result in. If not provided, a new array is allocated.

        Returns
        -------
        out : 1d np.ndarray
            Calculated velocity spectrum values for a specified `velocity` in time range from `t_min_ix` to `t_max_ix`.
        """
        t_win_size_min_ix = max(0, t_min_ix - half_win_size_samples)
        t_win_size_max_ix = min(len(times) - 1, t_max_ix + half_win_size_samples)

        if correction_type == 'LMO':
            corrected_gather_data = apply_constant_velocity_lmo(gather_data, offsets, sample_interval, delay,
                                                    times[t_win_size_min_ix: t_win_size_max_ix + 1], velocity,
                                                    interpolate)
        elif correction_type == 'NMO':
            corrected_gather_data = apply_constant_velocity_nmo(gather_data, offsets, sample_interval, delay,
                                                    times[t_win_size_min_ix: t_win_size_max_ix + 1], velocity,
                                                    max_strecth_factor, interpolate)
        else:
            raise ValueError('correction_type should be either "NMO" or "LMO"')

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
    def _calc_spectrum_numba(spectrum_func, coherency_func, correction_type, gather_data, times, offsets, velocities,
                             sample_interval, delay, half_win_size_samples, interpolate, max_strecth_factor):
        """Parallelized and njitted method for velocity spectrum calculation.

        Parameters
        ----------
        spectrum_func : njitted callable
            Base function for velocity spectrum calculation for a single velocity and a time range.
        coherency_func : njitted callable
            A function for hodograph coherency estimation.
        correction_type: str, 'LMO' or 'NMO'
            Type of correction to perform. 'NMO' for normal moveout, 'LMO' for linear moveout.
        other parameters : misc
            Passed directly from class attributes or `from_gather` arguments (except for `velocities`
            which are converted from m/s to m/ms).

        Returns
        -------
        velocity_spectrum : 2d np.ndarray
            Array with velocity spectrum values.
        """
        velocity_spectrum = np.empty((gather_data.shape[1], len(velocities)), dtype=np.float32)
        for j in prange(len(velocities)):  # pylint: disable=consider-using-enumerate
            spectrum_func(coherency_func=coherency_func, correction_type=correction_type, gather_data=gather_data,
                          times=times, offsets=offsets, velocity=velocities[j], sample_interval=sample_interval,
                          delay=delay, half_win_size_samples=half_win_size_samples, t_min_ix=0,
                          t_max_ix=gather_data.shape[1], interpolate=interpolate,
                          max_strecth_factor=max_strecth_factor, out=velocity_spectrum[:, j])
        return velocity_spectrum


class VerticalVelocitySpectrum(BaseVelocitySpectrum):
    r"""A class for Vertical Velocity Spectrum calculation and processing.

    Vertical velocity spectrum is a measure of hodograph coherency. The higher the values of velocity spectrum are,
    the more coherent the signal is along a hyperbolic trajectory over the spread length of the gather.

    Velocity spectrum instance can be created:
    1. Directly by passing spectrum values, times and velocities to `__init__` .
    2. By passing the gather (and optional parameters such as velocity range, window size, coherency measure
       and a factor for stretch mute) to `from_gather` constructor.
    3. By calling :func:`~Gather.calculate_vertical_velocity_spectrum` method (recommended way).

    To calculate velocity spectrum from gather:
        :math:`VS(k, v) = \frac{\sum^{k+N/2}_{i=k-N/2} numerator(i, v)}
                            {\sum^{k+N/2}_{i=k-N/2} denominator(i, v)},

    where:
     - VS - velocity spectrum value for starting time index `k` and velocity `v`,
     - N - temporal window size,
     - numerator(i, v) - numerator of the coherency measure,
     - denominator(i, v) - denominator of the coherency measure.

    For different coherency measures the numerator and denominator are calculated as follows:

    - Stacked Amplitude, "S":
        numerator(i, v) = abs(sum^{M-1}_{j=0} f_{j}(i, v))
        denominator(i, v) = 1

    - Normalized Stacked Amplitude, "NS":
        numerator(i, v) = abs(sum^{M-1}_{j=0} f_{j}(i, v))
        denominator(i, v) = sum^{M-1}_{j=0} abs(f_{j}(i, v))

    - Semblance, "NE":
        numerator(i, v) = (sum^{M-1}_{j=0} f_{j}(i, v))^2 / M
        denominator(i, v) = sum^{M-1}_{j=0} f_{j}(i, v)^2

    - Crosscorrelation, "CC":
        numerator(i, v) = ((sum^{M-1}_{j=0} f_{j}(i, v))^2 - sum^{M-1}_{j=0} f_{j}(i, v)^2) / 2
        denominator(i, v) = 1

    - Energy Normalized Crosscorrelation, "ENCC":
        numerator(i, v) = ((sum^{M-1}_{j=0} f_{j}(i, v))^2 - sum^{M-1}_{j=0} f_{j}(i, v)^2) / (M - 1)
        denominator(i, v) = sum^{M-1}_{j=0} f_{j}(i, v)^2

    where f_{j}(i, v) is the amplitude value on the `j`-th trace being NMO-corrected for time index `i` and velocity
    `v`. Thus the amplitude is taken for the time defined by :math:`t(i, v) = \sqrt{t_0^2 + \frac{l_j^2}{v^2}}`, where:
    :math:`t_0` - start time of the hyperbola associated with time index `i`,
    :math:`l_j` - offset of the `j`-th trace,
    :math:`v` - velocity value.

    See the COHERENCY_FUNCS for the full list of available coherency measures.

    The resulting matrix :math:`VS(k, v)` has shape (n_times, n_velocities) and contains vertical velocity spectrum
    values based on hyperbolas with each combination of the starting point :math:`k` and velocity :math:`v`.

    The algorithm for velocity spectrum calculation looks as follows:
    For each velocity from the given velocity range:
        1. Calculate NMO-corrected gather.
        2. Estimate numerator and denominator for given coherency measure for each timestamp.
        3. Get the values of velocity spectrum as a ratio of rolling sums of numerator and denominator in temporal
        windows of a given size.

    Examples
    --------
    Calculate velocity spectrum for 200 velocities from 2000 to 6000 m/s and a temporal window size of 16 ms:
    >>> survey = Survey(path, header_index=["INLINE_3D", "CROSSLINE_3D"], header_cols="offset")
    >>> gather = survey.sample_gather()
    >>> spectrum = gather.calculate_vertical_velocity_spectrum(velocities=np.linspace(2000, 6000, 200), window_size=16)

    Parameters
    ----------
    spectrum : 2d np.ndarray
        An array with vertical velocity spectrum values.
    velocities : 1d np.ndarray
        Stacking velocity values corresponding to the velocity spectrum. Measured in meters/seconds.
    times: 1d np.ndarray
        Timestamps corresponding to the velocity spectrum. Measured in milliseconds.
    gather : Gather, optional, defaults to None
        Seismic gather corresponding to the velocity spectrum.
    coords : Coordinates, optional, defaults to None
        Spatial coordinates of the velocity spectrum.

    Attributes
    ----------
    spectrum : 2d np.ndarray
        An array with vertical velocity spectrum values.
    velocities : 1d np.ndarray
        Stacking velocity values corresponding to the velocity spectrum. Measured in meters/seconds.
    times: 1d np.ndarray
        Timestamps corresponding to the velocity spectrum. Measured in miliseconds.
    gather : Gather or None
        Seismic gather corresponding to the velocity spectrum.
    coords : Coordinates or None
        Spatial coordinates of the velocity spectrum.
    stacking_velocity : StackingVelocity or None
        Stacking velocity around which velocity spectrum was calculated.
    relative_margin : float or None
        Relative margin for which velocity range obtained from `stacking_velocity` was additionally extended.
    coherency_func : callable or None
        A function that estimates the hodograph's coherency measure.
    half_win_size_samples : int or None
        Half of the temporal window size for smoothing the velocity spectrum. Measured in samples.
    max_stretch_factor : float or np.inf
        Maximum allowable factor for stretch muter.
    correction_type: 'NMO'
        Gather moveout correction method used for velocity spectrum computation.
    """
    correction_type = 'NMO'

    def __init__(self, spectrum, velocities, times, gather=None, coords=None):
        super().__init__(spectrum, velocities, times, gather=gather, coords=coords)

        # Set attributes relative to instance created `from_gather `.
        self.stacking_velocity = None
        self.relative_margin = None
        self.coherency_func = None
        self.half_win_size_samples = None
        self.max_stretch_factor = np.inf

    @property
    def velocities(self):
        """np.ndarray: Array of spectrum's velocity values. Measured in meters/seconds."""
        return self.x_values

    @property
    def n_velocities(self):
        """int: The number of velocities the spectrum was calculated for."""
        return len(self.velocities)

    @classmethod
    def from_gather(cls, gather, velocities=None, stacking_velocity=None, relative_margin=0.2, velocity_step=50,
                    window_size=50, mode='semblance', max_stretch_factor=np.inf, interpolate=True):
        r"""Calculate Vertical Velocity Spectrum from gather.
        The detailed description of computation algorithm can be found in the class docs.

        Parameters
        ----------
        gather : Gather
            Seismic gather to calculate velocity spectrum for.
        velocities : 1d np.ndarray, optional, defaults to None
            An array of stacking velocities to calculate the velocity spectrum for. Measured in meters/seconds. If not
            provided, `stacking_velocity` is evaluated for gather times to estimate the velocity range being examined.
            The resulting velocities are then evenly sampled from this range being additionally extended by
            `relative_margin` * 100% in both directions with a step of `velocity_step`.
        stacking_velocity : StackingVelocity or StackingVelocityField, optional, defaults to DEFAULT_STACKING_VELOCITY
            Stacking velocity around which vertical velocity spectrum is calculated if `velocities` are not given.
            `StackingVelocity` instance is used directly. If `StackingVelocityField` instance is passed, a
            `StackingVelocity` corresponding to gather coordinates is fetched from it.
        relative_margin : float, optional, defaults to 0.2
            Relative velocity margin to additionally extend the velocity range obtained from `stacking_velocity`: an
            interval [`min_velocity`, `max_velocity`] is mapped to [(1 - `relative_margin`) * `min_velocity`,
            (1 + `relative_margin`) * `max_velocity`].
        velocity_step : float, optional, defaults to 50
            A step between two adjacent velocities for which vertical velocity spectrum is calculated if `velocities`
            are not passed. Measured in meters/seconds.
        window_size : int, optional, defaults to 50
            Temporal window size used for velocity spectrum calculation. The higher the `window_size` is, the smoother
            the resulting velocity spectrum will be but to the detriment of small details. Measured in milliseconds.
        mode: str, optional, defaults to 'semblance'
            The measure for estimating hodograph coherency.
            The available options are:
                `semblance` or `NE`,
                `stacked_amplitude` or `S`,
                `normalized_stacked_amplitude` or `NS`,
                `crosscorrelation` or `CC`,
                `energy_normalized_crosscorrelation` or `ENCC`.
        max_stretch_factor : float, defaults to np.inf
            Maximum allowable factor for the muter that attenuates the effect of waveform stretching after
            NMO correction. This mute is applied after NMO correction for each provided velocity and before coherency
            calculation. The lower the value, the stronger the mute. In case np.inf (default) no mute is applied.
            Reasonably good value is 0.65.
        interpolate: bool, optional, defaults to True
            Whether to perform linear interpolation to retrieve amplitudes along hodographs. If `False`, an amplitude
            at the nearest time sample is used.

        Returns
        -------
        spectrum : VerticalVelocitySpectrum
            Vertical velocity spectrum instance.
        """
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
                  "half_win_size_samples": half_win_size_samples, "interpolate": interpolate,
                  "correction_type": cls.correction_type, "max_strecth_factor": max_stretch_factor}

        velocity_spectrum = cls._calc_spectrum_numba(**kwargs)
        spectrum = cls(velocity_spectrum, velocities, gather.times, coords=gather.coords, gather=gather.copy())

        spectrum.stacking_velocity = stacking_velocity
        spectrum.relative_margin = relative_margin
        spectrum.coherency_func = coherency_func
        spectrum.half_win_size_samples = half_win_size_samples
        spectrum.max_stretch_factor = max_stretch_factor
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
    def plot(self, stacking_velocity=None, *, interactive=False, title=None, **kwargs):
        """Plot vertical velocity spectrum.

        Parameters
        ----------
        stacking_velocity : StackingVelocity, StackingVelocityField or str, optional
            Stacking velocity to plot if given. If StackingVelocityField instance is passed,
            a StackingVelocity corresponding to spectrum coordinates is fetched from it.
            May be `str` if plotted in a pipeline: in this case it defines a component with stacking velocities to use.
        interactive : bool, optional, defaults to False
            Whether to plot velocity spectrum in interactive mode. This mode also plots the gather used to calculate
            the velocity spectrum. Clicking on velocity spectrum highlights the corresponding hodograph on the gather
            plot and allows performing NMO correction of the gather with the selected velocity.
            Interactive plotting must be performed in a JupyterLab environment with the `%matplotlib widget`
            magic executed and `ipympl` and `ipywidgets` libraries installed.
        title : str, optional
            Plot title. If not provided, equals to stacked lines "Vertical Velocity Spectrum" and coherency func name.
        gather_plot_kwargs : dict, optional, only for interactive mode
            Additional arguments to pass to `Gather.plot`.
        kwargs : misc, optional
            Additional common keyword arguments for `Spectrum.plot`.
        """
        if title is None:
            title = f"Vertical Velocity Spectrum \n Coherency func: {self.coherency_func.__name__}"
        if isinstance(stacking_velocity, StackingVelocityField):
            stacking_velocity = stacking_velocity(self.coords)

        plot_kwargs = {"vfunc": stacking_velocity, "title": title, "half_win_size": self.half_win_size_samples or 10,
                      "x_label": "Velocity, m/s", "y_label": 'Time, ms',  **kwargs}

        if not interactive:
            return super().plot(**plot_kwargs)
        return VerticalVelocitySpectrumPlot(self, **plot_kwargs).plot()

    @batch_method(target="for", args_to_unpack="init", copy_src=False)
    def calculate_stacking_velocity(self, init=None, bounds=None, relative_margin=None, acceleration_bounds="auto",
                                    times_step=100, max_offset=5000, hodograph_correction_step=25, max_n_skips=2):
        """Calculate stacking velocity by vertical velocity spectrum.

        Notes
        -----
        A detailed description of the proposed algorithm and its implementation can be found in
        :func:`~velocity_model.calculate_stacking_velocity` docs.

        Parameters
        ----------
        init : StackingVelocity or str, optional
            A rough estimate of the stacking velocity being picked. Used to calculate `bounds` as
            [`init` * (1 - `relative_margin`), `init` * (1 + `relative_margin`)] if they are not given.
            May be `str` if called in a pipeline: in this case it defines a component with stacking velocities to use.
            If not given, `self.stacking_velocity` is used.
        bounds : array-like of two StackingVelocity, optional
            Left and right bounds of an area for stacking velocity picking. If not given, `init` must be passed.
        relative_margin : positive float, optional
            A fraction of stacking velocities defined by `init` used to estimate `bounds` if they are not given.
            If not given, `self.relative_margin` is used.
        acceleration_bounds : tuple of two positive floats or "auto" or None, optional
            Minimal and maximal acceleration allowed for the stacking velocity function. If "auto", equals to the range
            of accelerations of stacking velocities in `bounds` extended by 50% in both directions. If `None`, only
            ensures that picked stacking velocity is monotonically increasing. Measured in meters/seconds^2.
        times_step : float, optional, defaults to 100
            A difference between two adjacent times defining graph nodes.
        max_offset : float, optional, defaults to 5000
            An offset for hodograph time estimation. Used to create graph nodes and calculate their velocities for each
            time.
        hodograph_correction_step : float, optional, defaults to 25
            The maximum difference in arrival time of two hodographs starting at the same zero-offset time and two
            adjacent velocities at `max_offset`. Used to create graph nodes and calculate their velocities for each
            time.
        max_n_skips : int, optional, defaults to 2
            Defines the maximum number of intermediate times between two nodes of the graph. Greater values increase
            computational costs, but tend to produce smoother stacking velocity.

        Returns
        -------
        stacking_velocity : StackingVelocity
            Calculated stacking velocity.
        """
        kwargs = {"init": get_first_defined(init, self.stacking_velocity), "bounds": bounds,
                  "relative_margin": get_first_defined(relative_margin, self.relative_margin),
                  "acceleration_bounds": acceleration_bounds, "times_step": times_step, "max_offset": max_offset,
                  "hodograph_correction_step": hodograph_correction_step, "max_n_skips": max_n_skips}
        return StackingVelocity.from_vertical_velocity_spectrum(self, **kwargs)


class ResidualVelocitySpectrum(BaseVelocitySpectrum):
    """A class for residual vertical velocity spectrum calculation and processing.
    Residual velocity spectrum is a hodograph coherency measure for a CDP gather along picked stacking velocity.

    Residual Velocity Spectrum instance can be created:
    1. Directly by passing spectrum values, times and margins to `__init__`.
    2. By passing the gather and stacking velocity (and optional parameters such as relative_margin)
       to `from_gather` constructor.
    3. By calling :func:`~Gather.calculate_residual_velocity_spectrum` method (recommended way).

    The method for residual spectrum computation from gather for a given time and velocity completely coincides with
    the calculation of :func:`~VerticalVelocitySpectrum.from_gather`, however, residual velocity spectrum is computed
    in a small area around givenstacking velocity, thus allowing for additional optimizations.

    The boundaries in which calculation is performed depend on time `t` and are given by:
    `stacking_velocity(t)` * (1 +- `relative_margin`).

    Since the length of this velocity range varies for different timestamps, the residual velocity spectrum values
    are interpolated to obtain a rectangular matrix of size (`n_times`, max(right_boundary - left_boundary)), where
    `left_boundary` and `right_boundary` are arrays of left and right boundaries for all timestamps respectively.

    Thus the residual velocity spectrum is a function of time and relative velocity margin. Zero margin line
    corresponds to the given stacking velocity and generally should pass through local velocity spectrum maxima.

    Examples
    --------
    First let's sample a CDP gather from a survey:
    >>> survey = Survey(path, header_index=["INLINE_3D", "CROSSLINE_3D"], header_cols="offset")
    >>> gather = survey.sample_gather()

    Now let's calculate stacking velocity by velocity spectrum of the gather:
    >>> velocity_spectrum = gather.calculate_vertical_velocity_spectrum()
    >>> velocity = velocity_spectrum.calculate_stacking_velocity()

    Residual velocity spectrum for the gather and calculated stacking velocity can be obtained as follows:
    >>> residual_spectrum = gather.calculate_residual_velocity_spectrum(velocity)

    Parameters
    ----------
    spectrum : 2d np.ndarray
        An array with residual vertical velocity spectrum values.
    margins : 1d np.ndarray
        Velocity margins corresponding to the residual velocity spectrum.
    times: 1d np.ndarray
        Timestamps corresponding to the residual velocity spectrum. Measured in miliseconds.
    gather : Gather or None
        Seismic gather corresponding to the residual velocity spectrum.
    coords : Coordinates or None
        Spatial coordinates of the residual velocity spectrum.

    Attributes
    ----------
    spectrum : 2d np.ndarray
        An array with residual velocity spectrum values.
    margins : 1d np.ndarray
        An array of residul velocity spectrum margins.
    times: 1d np.ndarray
        An array of residul velocity spectrum timestamps. Measured in miliseconds.
    gather : Gather or None
        Seismic gather corresponding to the residual velocity spectrum.
    coords : Coordinates or None
        Spatial coordinates of the residual velocity spectrum.
    stacking_velocity : StackingVelocity
        Stacking velocity around which residual velocity spectrum was calculated.
    coherency_func : callable or None
        A function that estimates the chosen coherency measure for a hodograph.
    half_win_size_samples : int
        Half of the temporal window size for smoothing the velocity spectrum. Measured in samples.
    max_stretch_factor: float
        Maximum allowable factor for stretch muter.
    correction_type: 'NMO'
        Gather correction method used for spectrum computation.
    """
    correction_type = 'NMO'

    def __init__(self, spectrum, margins, times, gather=None, coords=None):
        super().__init__(spectrum, margins, times, gather=gather, coords=coords)

        # Set attributes relative to instance created `from_gather`.
        self.stacking_velocity = None
        self.coherency_func = None
        self.half_win_size_samples = None
        self.max_stretch_factor = np.inf

    @property
    def margins(self):
        """np.ndarray: Array of residual spectrum's velocity margins."""
        return self.x_values

    @property
    def n_margins(self):
        """int: The number of velocity margins the spectrum was calculated for."""
        return len(self.margins)

    @classmethod
    def from_gather(cls, gather, stacking_velocity, relative_margin=0.2, velocity_step=25, window_size=50,
                    mode='semblance', max_stretch_factor=np.inf, interpolate=True):
        """Calculate Residual Velocity Spectrum from gather.
        The description of computation algorithm can be found in the class docs.

        Parameters
        ----------
        gather : Gather
            Seismic gather to calculate residual velocity spectrum for.
        stacking_velocity : StackingVelocity or StackingVelocityField
            Stacking velocity around which residual velocity spectrum to calculate. `StackingVelocity` instance is used
            directly. If `StackingVelocityField` instance is passed, a `StackingVelocity` corresponding to gather
            coordinates is fetched from it.
        relative_margin : float, optional, defaults to 0.2
            Relative velocity margin, that determines the velocity range for velocity spectrum calculation for each time
            `t` as `stacking_velocity(t)` * (1 +- `relative_margin`).
        velocity_step : float, optional, defaults to 25
            A step between two adjacent velocities for which residual velocity spectrum is calculated. Measured in
            meters/seconds.
        window_size : int, optional, defaults to 50
            Temporal window size used for velocity spectrum calculation. The higher the `window_size` is, the smoother
            the resulting velocity spectrum will be but to the detriment of small details. Measured in milliseconds.
        mode: str, optional, defaults to 'semblance'
            The measure for estimating hodograph coherency.
            The available options are:
                `semblance` or `NE`,
                `stacked_amplitude` or `S`,
                `normalized_stacked_amplitude` or `NS`,
                `crosscorrelation` or `CC`,
                `energy_normalized_crosscorrelation` or `ENCC`.
        max_stretch_factor : float, defaults to np.inf
            Maximum allowable factor for the muter that attenuates the effect of waveform stretching  after
            NMO correction. This mute is applied after NMO correction for each provided velocity and before coherency
            calculation. The lower the value, the stronger the mute. In case np.inf (default) no mute is applied.
            Reasonably good value is 0.65.
        interpolate: bool, optional, defaults to True
            Whether to perform linear interpolation to retrieve amplitudes along hodographs. If `False`, an amplitude at
            the nearest time sample is used.

        Returns
        -------
        spectrum : ResidualVelocitySpectrum
            Residual velocity spectrum instance.
        """
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
                  "half_win_size_samples": half_win_size_samples, "interpolate": interpolate,
                  "correction_type": cls.correction_type, "max_strecth_factor": max_stretch_factor}


        velocity_spectrum = cls._calc_spectrum_numba(**kwargs)
        margins = np.linspace(-relative_margin, relative_margin, velocity_spectrum.shape[1])
        spectrum = cls(velocity_spectrum, margins, gather.times, coords=gather.coords, gather=gather.copy())

        spectrum.stacking_velocity = stacking_velocity.copy().crop(gather.times[0], gather.times[-1])
        spectrum.coherency_func = coherency_func
        spectrum.half_win_size_samples = half_win_size_samples
        spectrum.max_stretch_factor = max_stretch_factor
        return spectrum

    @staticmethod
    @njit(nogil=True, fastmath=True, parallel=True)
    def _calc_spectrum_numba(spectrum_func, coherency_func, correction_type, gather_data, times, offsets,
                             stacking_velocities, relative_margin, velocity_step, sample_interval, delay,
                             half_win_size_samples, interpolate, max_strecth_factor):
        """Parallelized and njitted method for residual vertical velocity spectrum calculation.

        Parameters
        ----------
        spectrum_func : njitted callable
            Base function for velocity spectrum calculation for a single velocity and a time range.
        coherency_func : njitted callable
            A function for hodograph coherency estimation.
        other parameters : misc
            Passed directly from class attributes or `__init__` arguments (except for `stacking_velocities` which are
            the values of `stacking_velocity` evaluated at gather times).

        Returns
        -------
        residual_velocity_spectrum : 2d np.ndarray
            Array with residual vertical velocity spectrum values.
        """

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
                          correction_type=correction_type, max_strecth_factor=max_strecth_factor,
                          interpolate=interpolate, out=velocity_spectrum[t_min_ix : t_max_ix + 1, i])

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
    def plot(self, *, acceptable_margin=None, title=None, interactive=False, **kwargs):
        """Plot residual vertical velocity spectrum. The plot always has a vertical line in the middle, representing
        the stacking velocity it was calculated for.

        Parameters
        ----------
        acceptable_margin : float, optional
            Defines an area around central stacking velocity that will be highlighted on the spectrum plot as
            `stacking_velocity(t)` * (1 +- `acceptable_margin`) for each time `t`. May be used for visual quality
            control of stacking velocity picking by setting this value low enough and checking that local maximas of
            velocity spectrum corresponding to primaries lie inside the highlighted area.
        title : str, optional
            Plot title. If not provided, equals to stacked lines "Residual Velocity Spectrum" and coherency func name.
        interactive : bool, optional, defaults to False
            Whether to plot residual velocity spectrum in interactive mode. This mode also plots the gather used to
            calculate the residual velocity spectrum. Clicking on residual velocity spectrum highlights the
            corresponding hodograph on the gather plot and allows performing NMO correction of the gather with the
            selected velocity. Interactive plotting must be performed in a JupyterLab environment with the
            `%matplotlib widget` magic executed and `ipympl` and `ipywidgets` libraries installed.
        gather_plot_kwargs : dict, optional, only for interactive mode
            Additional arguments to pass to `Gather.plot`.
        kwargs : misc, optional
            Additional common keyword arguments for `Spectrum.plot`.
        """
        if title is None:
            title = f"Residual Velocity Spectrum \n Coherency func: {self.coherency_func.__name__}"
        stacking_velocity = VFUNC(self.stacking_velocity.times, [0] * len(self.stacking_velocity.times))

        if acceptable_margin is not None:
            stacking_velocity.bounds = [VFUNC([0, self.samples[-1]], [-acceptable_margin, -acceptable_margin]),
                                        VFUNC([0, self.samples[-1]], [acceptable_margin, acceptable_margin])]

        plot_kwargs = {"vfunc": stacking_velocity, "title": title, "half_win_size": self.half_win_size_samples or 10,
                       "x_ticker": {"round_to": 2}, "x_label": "Margin", "y_label": 'Time, ms', **kwargs}

        if not interactive:
            return super().plot(**plot_kwargs)
        return RedidualVelocitySpectrumPlot(self, **plot_kwargs).plot()


class SlantStack(BaseVelocitySpectrum):
    """A class for Slant Stack calculation.

    Slant Stacking is a procedure of a plane-wave decomposition of a seismic gather.
    It can be achieved by applying linear moveout and summing amplitudes over the offset axis.
    Note that Slant Stack exist in time-velocity domain, not in conventional time-slowness.

    Slant Stack instance can be created:
    1. Directly by passing Slant Stack values, times and velocities to `__init__`.
    2. By passing the gather to `from_gather` constructor.
    3. By calling :func:`~Gather.calculate_slant_stack` method (recommended way).

    Parameters
    ----------
    slant_stack : 2d np.ndarray
        An array with slant stack values.
    velocities : 1d np.ndarray
        Velocities corresponding to the slant stack. Measured in meters/seconds.
    times: 1d np.ndarray
        Timestamps corresponding to the slant stack. Measured in miliseconds.
    gather : Gather, optional, defaults to None
        Seismic gather corresponding to the slant stack.
    coords : Coordinates, optional, defaults to None
        Spatial coordinates of the slant stack.

    Attributes
    ----------
    spectrum : 2d np.ndarray
        An array with slant stack values.
    velocities : 1d np.ndarray
        Velocities corresponding to the slant stack. Measured in meters/seconds.
    times: 1d np.ndarray
        Timestamps corresponding to the slant stack. Measured in miliseconds.
    gather : Gather or None
        Seismic gather corresponding to the slant stack.
    coords : Coordinates or None
        Spatial coordinates of the slant stack.
    correction_type: 'LMO'
        Gather correction method used for spectrum computation.
    """
    correction_type = 'LMO'

    def __init__(self, slant_stack, velocities, times, gather=None, coords=None):
        super().__init__(slant_stack, velocities, times, gather=gather, coords=coords)

    @property
    def velocities(self):
        """np.ndarray: Array of spectrum's velocity values. Measured in meters/seconds."""
        return self.x_values

    @classmethod
    def from_gather(cls, gather, velocities=None):
        """Calculate Slant Stack transform from gather.

        The method for slant stack computation for a given time and velocity coincides with
        the calculation of :func:`~VerticalVelocitySpectrum.from_gather` and looks as follows:

        For each velocity from the given velocity range:
            1. Calculate LMO-corrected gather.
            2. Estimate numerator and denominator for `stacked_amplitude_sum` coherency measure for each timestamp.
            3. Get the slant stack value as a ratio of numerator and denominator.

        Parameters
        ----------
        gather : Gather
            Seismic gather to calculate slant stack for.
        velocities : 1d np.ndarray, optional, defaults to None
            An array of stacking velocities to calculate the slant stack for. Measured in meters/seconds.
            If not provided, uniformly covers the range from 100 m/s to 2400 m/s with step 50 m/s.

        Returns
        -------
        spectrum : SlantStack
            SlantStack instance.
        """
        if velocities is None:
            velocities = np.arange(100, 2400, 50, dtype=np.float32)
        else:
            velocities = np.sort(velocities)
            velocities = np.asarray(velocities, dtype=np.float32)
        velocities_ms = velocities / 1000  # from m/s to m/ms

        kwargs = {"spectrum_func": cls.calc_single_velocity_spectrum,
                  "coherency_func": coherency_funcs.stacked_amplitude_sum,
                  "gather_data": gather.data, "times": gather.times, "offsets": gather.offsets,
                  "velocities": velocities_ms, "sample_interval": gather.sample_interval, "delay": gather.delay,
                  "half_win_size_samples": 0, "interpolate": True, "correction_type": cls.correction_type,
                  "max_strecth_factor": np.inf}

        velocity_spectrum = cls._calc_spectrum_numba(**kwargs)
        spectrum = cls(velocity_spectrum, velocities, gather.times, coords=gather.coords, gather=gather.copy())
        return spectrum

    @plotter(figsize=(10, 9))
    def plot(self, interactive=False, title=None, **kwargs):
        """Plot Slant Stack.

        Parameters
        ----------
        interactive : bool, optional, defaults to False
            Whether to plot slant stack in interactive mode. This mode also plots the gather used to calculate
            the slant stack. Clicking on slant stack highlights the corresponding hodograph on the gather
            plot and allows performing LMO correction of the gather with the selected velocity.
            Interactive plotting must be performed in a JupyterLab environment with the `%matplotlib widget`
            magic executed and `ipympl` and `ipywidgets` libraries installed.
        title : str, optional
            Plot title. If not provided, equals to 'Slant Stack'.
        gather_plot_kwargs : dict, optional, only for interactive mode
            Additional arguments to pass to `Gather.plot`.
        kwargs : misc, optional
            Additional common keyword arguments for `Spectrum.plot`.
        """
        if title is None:
            title = "Slant Stack"

        plot_kwargs = {"title": title, "x_label": "Velocity, m/s", "y_label": 'Time, ms', **kwargs}

        if not interactive:
            return super().plot(**plot_kwargs)
        return SlantStackPlot(self, **plot_kwargs).plot()
