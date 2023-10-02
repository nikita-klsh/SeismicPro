import numpy as np
from numba import njit, prange

from .dispersion_curve import DispersionCurve
from .utils.bessel import j0, y0
from ..spectrum import Spectrum
from ..velocity_spectrum import SlantStack
from ..decorators import plotter, batch_method
from ..containers import SamplesContainer
from .interactive_plot import DispersionSpectrumPlot


class DispersionSpectrum(Spectrum):
    """ Implements various transforms of seisimc gather to f-v domain. """

    @classmethod
    def from_gather(cls, gather, velocities, fmax=None, spectrum_type='fv', complex_to_real=np.abs, start=None, end=None, **kwargs):
        n_df = kwargs.pop('n_df', 1)
        if spectrum_type == 'fv':
            slant_stack = SlantStack.from_gather(gather, velocities)
            return  cls.from_slant_stack(slant_stack, fmax, complex_to_real, **kwargs)
        else:
            ft, frequencies = cls.calculate_ft(gather.data, gather.sample_interval / 1000, fmax, n_df=n_df)

    
        WAVEFIELD_TRANSFORMS = {
            'phase_shift': cls.calculate_ps,
            'beam_former': cls.calculate_fdbf
        }

        spectrum_func = WAVEFIELD_TRANSFORMS.get(spectrum_type)
        
        if start is not None:
            start_offsets = start(frequencies)
        else:
            start_offsets = np.full_like(frequencies, 0, dtype=np.float64)

        if end is not None:
            end_offsets = end(frequencies)
        else:
            end_offsets = np.full_like(frequencies, 1e6, dtype=np.float64)
        
        spectrum_data = spectrum_func(ft, velocities, frequencies, gather.offsets, start=start_offsets, end=end_offsets, **kwargs)
        spectrum_data = complex_to_real(spectrum_data)
        spectrum =  cls(spectrum_data, velocities, frequencies)
        spectrum.times_interval =  gather.sample_rate / gather.n_samples / n_df
        spectrum.n_df = n_df
        spectrum.delay = frequencies[0]
        spectrum.gather = gather.copy()
        spectrum.coords = gather.coords
        spectrum.start = start
        spectrum.end = end
        return spectrum

    @property
    def velocities(self):
        return self.x_values

    @property
    def n_velocities(self):
        return len(self.velocities)

    @property
    def frequencies(self):
        return self.y_values

    def plot(self, dispersion_curve=None, plot_bounds=True, interactive=False, **kwargs):
        if not interactive:
            return super().plot(title='Dispersion Spsectrum', vfunc=dispersion_curve, plot_bounds=plot_bounds, align_vfunc=False, **kwargs)
        return DispersionSpectrumPlot(self, title='Dispersion Spsectrum', vfunc=dispersion_curve, plot_bounds=plot_bounds, half_win_size=1, **kwargs).plot()

    @staticmethod
    def calculate_ft(data, sample_interval, fmax=None, n_df=1):
        """Perform 1d Fourier transform of given 2d array of signals along the 1 axis.
        Transform is done for frequencies not greater than fmax.
        Returns 2d array of transform and array of corresponding frequencies. """
        ft_data = np.fft.fft(data, data.shape[1] * n_df, axis=1)
        frequencies = np.fft.fftfreq(data.shape[1] * n_df, sample_interval)

        max_frequency = fmax or frequencies.max()
        frequencies_mask = (frequencies > 0) & (frequencies <= max_frequency)
        return ft_data[:, frequencies_mask], frequencies[frequencies_mask]


    @classmethod
    def from_slant_stack(cls, slant_stack, fmax=None, complex_to_real=np.abs):
        spectrum_data, frequencies = cls.calculate_ft(slant_stack.spectrum.T, slant_stack.gather.sample_interval / 1000, fmax)
        spectrum_data = complex_to_real(spectrum_data).T
        spectrum =  cls(spectrum_data, slant_stack.x_values, frequencies)
        spectrum.gather = slant_stack.gather.copy()
        spectrum.times_interval = slant_stack.gather.sample_rate / slant_stack.gather.n_samples
        spectrum.delay = frequencies[0]
        spectrum.coords = slant_stack.gather.coords
        return spectrum

    @batch_method(target="for", args_to_unpack="init", copy_src=False)
    def calculate_dispersion_curve(self, fmin=None, fmax=None, init=None, bounds=None, relative_margin=0.2, velocity_step=10,
                                      acceleration_bounds="adaptive", times_step=None, max_n_skips=2):

        from .fields import DispersionField
        if isinstance(init, DispersionField):
            init = init(self.coords)

        if times_step is None:
            times_step = self.times_interval

        fmin = fmin or self.frequencies.min()
        fmax = fmax or self.frequencies.max()
        mask = (self.frequencies >= fmin) & (self.frequencies <= fmax)

        spectrum =  type(self)(self.spectrum[mask], self.velocities, self.frequencies[mask])
        spectrum.times_interval = self.times_interval
        spectrum.delay = self.delay
        spectrum.gather = self.gather.copy()
        spectrum.coords = self.gather.coords
        return DispersionCurve.from_dispersion_spectrum(spectrum, init=init, bounds=bounds, relative_margin=relative_margin, velocity_step=velocity_step,
                                       acceleration_bounds=acceleration_bounds, times_step=times_step, max_n_skips=max_n_skips)


    @staticmethod
    @njit(parallel=True)
    def calculate_ps(ft_gather_data, velocities, frequencies, offsets, start=None, end=None):
        if start is None:
            start =  np.full_like(frequencies, 0, dtype=np.float64)
        if end is None:
            end = np.full_like(frequencies, 1e5, dtype=np.float64)

        masks = np.full(ft_gather_data.T.shape, True, dtype=np.bool_)
        for col in range(len(frequencies)):
            masks[col] = (offsets <= end[col]) & (offsets >= start[col])

        spectrum_data = np.empty((len(velocities), len(frequencies)), dtype=np.complex64)
        ft_gather_data = ft_gather_data / np.abs(ft_gather_data)
        for row in prange(len(velocities)):
            velocity = velocities[row]
            delta = offsets / velocity
            for col in range(len(frequencies)):
                mask = masks[col]
                if mask.sum() == 0:
                    spectrum_data[row, col] = 0
                else:
                    frequency = frequencies[col]
                    shift = np.exp(1j * 2* np.pi * frequency * delta[mask])
                    inner = shift * ft_gather_data[mask, col]
                    spectrum_data[row, col] = np.mean(inner)
        return spectrum_data.T


    @staticmethod
    @njit(parallel=True)
    def calculate_fdbf(ft_gather_data, velocities, frequencies, offsets, cylindrical=True, weighted=True, start=None, end=None):
        if start is None:
            start =  np.full_like(frequencies, offsets.min(), dtype=np.float64)
        if end is None:
            end = np.full_like(frequencies, offsets.max(), dtype=np.float64)    
        spectrum_data = np.empty((len(frequencies), len(velocities)), dtype=np.complex64)
        if weighted:
            a, b = np.histogram(offsets, bins=100)
            ix = np.searchsorted(b, offsets)
            ix[ix == 0] = 1
            ix = ix  - 1
            a[a == 0] = 1
            w = (1 / a[ix]).astype(np.float32)
        else:
            w = np.full_like(offsets, 1, np.float32)

        k = 2 * np.pi * frequencies.reshape(-1, 1) / velocities.reshape(1, -1)
        for i in prange(len(frequencies)):
            mask = (offsets <= end[i]) & (offsets >= start[i])
            for j in range(len(velocities)):
                kx = k[i, j] * offsets

                if cylindrical:
                    h0_kx = j0(kx) + 1j * y0(kx)
                    angle_kx = np.angle(h0_kx)
                    steer = np.exp(-1j * angle_kx).reshape(1, -1)
                else:
                    steer = np.exp(-1j * kx).reshape(1, -1)

                steer = w * steer

                HS = np.conjugate(steer)[:, mask] @ ft_gather_data[:, i][mask]
                spectrum_data[i, j] = (HS * np.conjugate(HS).T).item() / len(mask)
        return spectrum_data
