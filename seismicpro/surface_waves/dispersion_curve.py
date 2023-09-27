from copy import copy
from functools import partial

import numpy as np
from disba import surf96
from scipy.optimize import minimize

from ..utils import VFUNC
from ..decorators import batch_method

from ..stacking_velocity.velocity_model import calculate_stacking_velocity


class DispersionCurve(VFUNC):

    def __init__(self, frequencies, velocities, coords=None):
        super().__init__(frequencies, velocities, coords)
        self.bounds = None

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

    @property
    def periods(self):
        return 1 / self.frequencies

    @classmethod
    def from_dispersion_curves(cls, velocities, weights=None, coords=None):
        return cls.from_vfuncs(velocities, weights, coords)
        
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


    @classmethod
    def from_elastic_model(cls, vs, f=None, dc=0.005):
        if f is not None:
            f = f
        elif f is None:
            if vs.produced_by is not None:
                f = vs.produced_by.frequencies
        else:
            raise ValueError('Provide frequencies range')
        v = cls.func(vs.velocities / 1000, 1 / f, vs.thickness / 1000, dc=dc, poison=vs.vpvs)
        return DispersionCurve(f, v * 1000)
    

    # @batch_method(target="for", copy_src=False)
    # def invert(self, fmin=None, fmax=None, dz=0.005, x0=None, bounds=None, vpvs=2, kd=2, alpha=0.005, dc=0.005, adaptive=False, tol=0.010):
    #     target_dispersion_curve = self.copy()
    #     target_dispersion_curve.filter(fmin, fmax)

    #     vs_law = VelocityLaw.from_dispersion_curve(target_dispersion_curve, kd)
    #     elevations = np.arange(dz, vs_law.depths.max() / 1000 + dz, dz)
    #     vs = vs_law(elevations * 1000) / 1000
    #     vp = vs * vpvs
    #     rho = vp * 0.32 + 0.77
    #     thickness = np.array([dz] * len(elevations))

    #     dv = 0.3
    #     boarders = np.random.choice([-1, 1], (len(vs), len(vs)))
    #     initial_simplex = np.concatenate([vs.reshape(1, -1), vs + dv * boarders], axis=0)
    #     initial_simplex = None

    #     loss = partial(self.loss, alpha=alpha, poison=vpvs)
    #     if x0 is None:
    #         x0 = vs
    #     if bounds is None:
    #         bounds = [(0.1, 5)] * len(vs)
        
    #     loss = partial(self.loss, alpha=alpha, poison=vpvs, dc=dc)
    #     scipy_res = minimize(loss, args=(target_dispersion_curve.velocities / 1000, target_dispersion_curve.periods, thickness), x0=x0, bounds=bounds, 
    #                           method="Nelder-Mead", tol=tol, options=dict(maxfev=2000, initial_simplex=initial_simplex, adaptive=adaptive))
    #     law = VelocityLaw(elevations * 1000, scipy_res.x * 1000, coords=self.coords)
    #     law.produced_by = target_dispersion_curve
    #     return law

    @batch_method(target="for", copy_src=False)
    def invert(self, fmin=None, fmax=None, dz=0.005, x0=None, bounds=None, vpvs=2, kd=2, alpha=0.005, dc=0.005, adaptive=False, tol=0.010, fit_vpvs=False):
        target_dispersion_curve = self.copy()
        target_dispersion_curve.filter(fmin, fmax)
        
        vs_law = VelocityLaw.from_dispersion_curve(target_dispersion_curve, kd)
        elevations = np.arange(dz, vs_law.depths.max() / 1000 + dz, dz)
        vs = vs_law(elevations * 1000) / 1000
        vp = vs * vpvs
        rho = vp * 0.32 + 0.77
        thickness = np.array([dz] * len(elevations))

        dv = 1
        boarders = np.random.choice([-1, 1], (len(vs), len(vs)))
        initial_simplex = np.concatenate([vs.reshape(1, -1), vs + dv * boarders], axis=0)
        initial_simplex = None

        if x0 is None:
            x0 = vs
        
        if bounds is None:
            bounds = [(0.1, 5)] * len(vs) 
            
        if fit_vpvs:
            x0 = np.concatenate([vs, [0]])
            bounds = [(0.1, 5)] * len(vs) + [(-1, 1)]

            boarders = np.random.choice([-1, 1], (len(vs) + 1, len(vs) + 1))
            some_vs = np.concatenate([vs, [0]])
            initial_simplex = np.concatenate([some_vs.reshape(1, -1), some_vs + dv * boarders], axis=0)

        loss = partial(self.loss, alpha=alpha, poison=vpvs, dc=dc, fit_vpvs=fit_vpvs)
        scipy_res = minimize(loss, args=(target_dispersion_curve.velocities / 1000, target_dispersion_curve.periods, thickness), x0=x0, bounds=bounds, 
                            method="Nelder-Mead", options=dict(maxfev=2000, initial_simplex=initial_simplex, adaptive=adaptive), tol=tol)
        
        if fit_vpvs:
            vs = scipy_res.x[:-1]
            vpvs = scipy_res.x[-1] + vpvs
        else:
            vs = scipy_res.x
            vpvs = vpvs
        
        law = VelocityLaw(elevations * 1000, vs * 1000, coords=self.coords)
        law.vpvs = vpvs
        law.produced_by = target_dispersion_curve
        return law


    # @staticmethod
    # def func(x, period, thickness, poison=2, dc=0.005):
    #     vs = x
    #     vp = vs * poison
    #     rho = vp * 0.32 + 0.77
    #     return surf96(period, thickness, vp, vs, rho, mode=0, itype=0, ifunc=3, dc=dc)


    @staticmethod
    def func(x, period, thickness, poison=2, dc=0.005, fit_vpvs=False):
        if not fit_vpvs:
            vs = x
            vp = vs * poison
        else:
            vs = x[:-1]
            vp = vs * (x[-1] + poison)
        
        rho = vp * 0.32 + 0.77
        return surf96(period, thickness, vp, vs, rho, mode=0, itype=0, ifunc=3, dc=dc)


    # @classmethod
    # def loss(cls, x, velocity, period, thickness, poison=2, dc=0.005, alpha=0.005):        
    #     try:
    #         return np.abs(velocity - cls.func(x, period, thickness, poison=poison, dc=dc)).mean() + alpha * np.abs(np.diff(x)).mean()
    #     except:
    #         return np.nan

    @classmethod
    def loss(cls, x, velocity, period, thickness, dc=0.005, alpha=0.005, fit_vpvs=False, poison=2):        
        if fit_vpvs:
            reg = np.abs(np.diff(x[:-1])).mean()
        else:
            reg = np.abs(np.diff(x)).mean()

        try:
            return np.abs(velocity - cls.func(x, period, thickness, dc=dc, poison=poison, fit_vpvs=fit_vpvs)).mean() + alpha * reg
        except:
            return np.nan


    
    def dump(self, path, n_decimals=2, encoding="UTF-8"):
        return super().dump(path, n_decimals=n_decimals, encoding=encoding)

        
class VelocityLaw(VFUNC):

    def __init__(self, depths, velocities, coords=None):
        super().__init__(depths, velocities, coords)
        self.vpvs = None

    @property
    def depths(self):
        """1d np.ndarray: An array with depth values for velocity laws. Measured in meters."""
        return self.data_x

    @property
    def velocities(self):
        """1d np.ndarray: An array with velocity values, matching the length of `depths`. Measured in meters/seconds."""
        return self.data_y

    @property
    def thickness(self):
        return np.diff(self.depths, prepend=0)

    @classmethod
    def from_dispersion_curve(cls, dispersion_curve, wavelenght_to_depath=2, rayleigh_to_shear=1.1):
        depths = dispersion_curve.wavelenghts / wavelenght_to_depath
        vs = dispersion_curve.velocities * rayleigh_to_shear
        ix = np.argsort(depths)
        law = cls(depths[ix], vs[ix])
        law.produced_by = dispersion_curve
        return law
