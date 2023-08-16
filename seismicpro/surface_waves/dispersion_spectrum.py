import numpy as np
from numba import njit, prange

from .dispersion_curve import DispersionCurve
from .utils.bessel import j0, y0
from ..spectrum import Spectrum
from ..velocity_spectrum import SlantStack



class DispersionSpectrum(Spectrum):
    """ Implements various transforms of seisimc gather to f-v domain. """

    @classmethod
    def from_gather(cls, gather, velocities, fmax=None, spectrum_type='fv', complex_to_real=np.abs, **kwargs):
        if spectrum_type == 'fv':
            slant_stack = SlantStack.from_gather(gather, velocities)
            return  cls.from_slant_stack(slant_stack, fmax, complex_to_real, **kwargs)

        ft, frequencies = cls.calculate_ft(gather.data, gather.sample_interval / 1000, fmax)
    
        WAVEFIELD_TRANSFORMS = {
            'phase_shift': cls.calculate_ps,
            'beam_former': cls.calculate_fdbf
        }

        spectrum_func = WAVEFIELD_TRANSFORMS.get(spectrum_type)
        spectrum_data = spectrum_func(ft, velocities, frequencies, gather.offsets, **kwargs)
        spectrum_data = complex_to_real(spectrum_data)
        spectrum =  cls(spectrum_data, velocities, frequencies)
        spectrum.sample_interval = gather.sample_rate / gather.n_samples
        spectrum.gather = gather.copy()
        return spectrum

    @staticmethod
    def calculate_ft(data, sample_interval, fmax=None):
        """Perform 1d Fourier transform of given 2d array of signals along the 1 axis.
        Transform is done for frequencies not greater than fmax.
        Returns 2d array of transform and array of corresponding frequencies. """
        ft_data = np.fft.fft(data, axis=1)
        frequencies = np.fft.fftfreq(data.shape[1], sample_interval)

        max_frequency = fmax or frequencies.max()
        frequencies_mask = (frequencies > 0) & (frequencies <= max_frequency)
        return ft_data[:, frequencies_mask], frequencies[frequencies_mask]


    @classmethod
    def from_slant_stack(cls, slant_stack, fmax=None, complex_to_real=np.abs):
        spectrum_data, frequencies = cls.calculate_ft(slant_stack.spectrum.T, slant_stack.gather.sample_interval / 1000, fmax)
        spectrum_data = complex_to_real(spectrum_data).T
        spectrum =  cls(spectrum_data, slant_stack.x_values, frequencies)
        spectrum.gather = slant_stack.gather.copy()
        return spectrum

    def calculate_dispersion_curve(self, init=None, bounds=None, relative_margin=0.2, velocity_step=10,
                                      acceleration_bounds="adaptive", times_step=100, max_n_skips=2):
        
        return DispersionCurve.from_dispersion_spectrum(self, init=init, bounds=bounds, relative_margin=relative_margin, velocity_step=velocity_step,
                                      acceleration_bounds=acceleration_bounds, times_step=times_step, max_n_skips=max_n_skips)


    @staticmethod
    @njit(parallel=True)
    def calculate_ps(ft_gather_data, velocities, frequencies, offsets):
        spectrum_data = np.empty((len(velocities), len(frequencies)), dtype=np.complex64)
        for row in prange(len(velocities)):
            velocity = velocities[row]
            delta = offsets / velocity
            for col in range(len(frequencies)):
                frequency = frequencies[col]
                shift = np.exp(1j * 2* np.pi * frequency * delta)
                inner = shift * ft_gather_data[:, col]
                spectrum_data[row, col] = np.sum(inner)
        return spectrum_data.T


    @staticmethod
    @njit(parallel=True)
    def calculate_fdbf(ft_gather_data, velocities, frequencies, offsets, cylindrical=True, weighted=True):
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
            for j in range(len(velocities)):
                kx = k[i, j] * offsets

                if cylindrical:
                    h0_kx = j0(kx) + 1j * y0(kx)
                    angle_kx = np.angle(h0_kx)
                    steer = np.exp(-1j * angle_kx).reshape(1, -1)
                else:
                    steer = np.exp(-1j * kx).reshape(1, -1)

                steer = w * steer

                HS = np.conjugate(steer) @ ft_gather_data[:, i]
                spectrum_data[i, j] = (HS * np.conjugate(HS).T).item()
        return spectrum_data
