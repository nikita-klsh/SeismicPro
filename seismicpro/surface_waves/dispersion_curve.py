from copy import copy

import numpy as np
from disba import surf96
from scipy.optimize import minimize

from ..utils import VFUNC
from ..decorators import batch_method

from ..stacking_velocity.velocity_model import calculate_stacking_velocity


class DispersionCurve(VFUNC):

    def __init__(self, frequencies, velocities, coords=None):
        super().__init__(frequencies, velocities, coords)

    @property
    def frequencies(self):
        """1d np.ndarray: An array with frequency values for which dispersion curve was estimated. Measured in HZ."""
        return self.data_x

    @property
    def velocities(self):
        """1d np.ndarray: An array with Rayleigh wave velocity values. Measured in meters/seconds."""
        return self.data_y

    @property
    def wavelenghts(self):
        """1d np.ndarray: An array with wavelengths. Measured in meters."""
        return self.velocities / self.frequencies

    
    @batch_method(target="for", args_to_unpack="init", copy_src=False)
    def invert(self, fmin=None, fmax=None, dz=0.005, bounds=(0.1, 5), vpvs=2.5, kd=2):
        elevations = np.arange(0, d.max() + dz, dz)
        
        target_dispersion_curve = self.copy().filter(fmin, fmax)
        vs_law = VelocityLaw.from_dispersion_curve(target_dispersion_curve, kd, vpvs)
        vs = vs_law(elevations)
        vp = vs * vpvs
        rho = vp * 0.32 + 0.77
        thickness = np.array([dz] * len(elevations))


        dv = 0.3
        boarders = np.random.choice([-1, 1], (len(vs), len(vs)))
        initial_simplex = np.concatenate([vs.reshape(1, -1), vs + dv * boarders], axis=0)

        scipy_res = minimize(cls.loss, args=(velocity, period, thickness, rho, vpvs), x0=vs, bounds=[bounds] * len(x0), 
                             method='Nelder-Mead', tol=0.010, options=dict(maxfev=2000, initial_simplex=initial_simplex))
    
        return VelocityLaw(elevations, scipy_res.x * 1000, coords=self.coords)
    
    @classmethod
    def from_dispersion_spectrum(cls, spectrum, init=None, bounds=None, relative_margin=0.2, velocity_step=10,
                                      acceleration_bounds="adaptive", times_step=100, max_n_skips=2):
        from .dispersion_spectrum import DispersionSpectrum  # pylint: disable=import-outside-toplevel
        if not isinstance(spectrum, DispersionSpectrum):
            raise ValueError("spectrum must be an instance of DispersionSPectrum")
        if init is None and bounds is None:
            raise ValueError("Either init or bounds must be passed")
        if init is not None and not isinstance(init, DispersionCurve):
            raise ValueError("init must be an instance of DispersionCurve")
        if bounds is not None:
            bounds = to_list(bounds)
            if len(bounds) != 2 or not all(isinstance(bound, DispersionCurve) for bound in bounds):
                raise ValueError("bounds must be an array-like with two DispersionCurve instances")

        kwargs = {"init": init, "bounds": bounds, "relative_margin": relative_margin,
                  "acceleration_bounds": acceleration_bounds, "max_n_skips": max_n_skips,
                  "times_step": times_step, "velocity_step": velocity_step}
                  
        stacking_velocity_params = calculate_stacking_velocity(spectrum, **kwargs)
        times, velocities, bounds_times, min_velocity_bound, max_velocity_bound = stacking_velocity_params
        coords = spectrum.coords  # Evaluate only once
        dispersion_curve = cls(times, velocities, coords=coords)
        dispersion_curve.bounds = [cls(bounds_times, min_velocity_bound, coords=coords),
                                    cls(bounds_times, max_velocity_bound, coords=coords)]
        return dispersion_curve

    def __call__(self, frequencies):
        """Return phase velocities for given `frequencies`.

        Parameters
        ----------
        frequencies : 1d array-like
            An array with time values. Measured in HZ.

        Returns
        -------
        velocities : 1d np.ndarray
            An array with phase velocity values, matching the length of `frequencies`. Measured in meters/seconds.
        """
        return np.maximum(super().__call__(frequencies), 0)


    @staticmethod
    def func(x, period, thickness, rho, poison=2, dc=0.005):
        return surf96(period, thickness, x * poison, x, rho, mode=0, itype=0, ifunc=3, dc=dc)


    @classmethod
    def loss(cls, x, velocity, period, thickness, rho, poison=2, dc=0.005, alpha=0.005):
        try:
            return np.abs(velocity - cls.func(x, period, thickness, rho, poison=poison, dc=dc)).mean() + alpha * np.abs(np.diff(x)).mean()
        except:
            return np.nan

        
class VelocityLaw(VFUNC):

    def __init__(depths, velocities, coords=None):
        super().__init__(depths, velocities, coords)

    def copy(self):
        return copy(self)

    @property
    def depths(self):
        """1d np.ndarray: An array with depth values for velocity laws. Measured in meters."""
        return self.data_x

    @property
    def velocities(self):
        """1d np.ndarray: An array with velocity values, matching the length of `depths`. Measured in meters/seconds."""
        return self.data_y

    @classmethod
    def from_dispersion_curve(cls, dispersion_curve, wavelenght_to_depath=2, rayleigh_to_shear=1.1):
        depths = dispersion_curve.wavelenghts / wavelenght_to_depath
        vs = dispersion_curve.velocities * rayleigh_to_shear

    def filter(self, fmin=None, fmax=None):
        fmin = fmin or self.frequencies.min()
        fmax = fmax or self.frequencies.max()
        mask = (self.frequencies >= fmin) & (self.frequencies <= fmax)
        self.frequencies = self.frequencies[mask]
        self.velocities = self.velocities[mask]
