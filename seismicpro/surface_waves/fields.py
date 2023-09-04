import os
from functools import partial, cached_property
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from .dispersion_curve import DispersionCurve, VelocityLaw
from ..field import SpatialField
from ..metrics import initialize_metrics
from ..utils import to_list, get_coords_cols, Coordinates, IDWInterpolator, ForPoolExecutor, GEOGRAPHIC_COORDS
from ..stacking_velocity import StackingVelocityField



class DispersionField(StackingVelocityField):

    item_class = DispersionCurve

    def construct_item(self, items, weights, coords):
        return self.item_class.from_dispersion_curves(items, weights, coords=coords)

    @cached_property
    def mean_dispersion_curve(self):
        return self.item_class.from_dispersion_curves(self.items)


class VSField(StackingVelocityField):
    
    item_class = VelocityLaw

    def construct_item(self, items, weights, coords):
        return self.item_class.from_vfuncs(items, weights, coords=coords)


    @cached_property
    def mean_velocity(self):
        return self.item_class.from_vfuncs(self.items)

    
    @staticmethod
    def _invert_dcs_together(dc_list, common_kwargs):
        """Fit a separate near-surface velocity model by offsets and times of first breaks for each set of parameters
        defined in `rv_kwargs_list`. This is a helper function and is defined as a `staticmethod` only to be picklable
        so that it can be passed to `ProcessPoolExecutor.submit`."""
        return [dc.invert(**common_kwargs) for dc in dc_list]
    
    
    @staticmethod
    def _invert_dcs_for(dc_list, common_kwargs):
        """Fit a separate near-surface velocity model by offsets and times of first breaks for each set of parameters
        defined in `rv_kwargs_list`. This is a helper function and is defined as a `staticmethod` only to be picklable
        so that it can be passed to `ProcessPoolExecutor.submit`."""
        return [dc.invert(**kwargs) for dc, kwargs in zip(dc_list, common_kwargs)]

    
    @classmethod
    def _invert_dc_parallel(cls, dc_list, common_kwargs=None, chunk_size=250, n_workers=None, bar=True, desc=None):
        """Fit a separate near-surface velocity model by offsets and times of first breaks for each set of parameters
        defined in `rv_kwargs_list`. Velocity model fitting is performed in parallel processes in chunks of size no
        more than `chunk_size`."""
        if common_kwargs is None:
            common_kwargs = {}
        n_velocities = len(dc_list)
        n_chunks, mod = divmod(n_velocities, chunk_size)
        if mod:
            n_chunks += 1
        if n_workers is None:
            n_workers = os.cpu_count()
        n_workers = min(n_chunks, n_workers)
        executor_class = ForPoolExecutor if n_workers == 1 else ProcessPoolExecutor

        futures = []
        with tqdm(total=n_velocities, desc=desc, disable=not bar) as pbar:
            with executor_class(max_workers=n_workers) as pool:
                for i in range(n_chunks):
                    chunk_kwargs = dc_list[i * chunk_size : (i + 1) * chunk_size]
                    chunk_common_kwargs = common_kwargs[i * chunk_size : (i + 1) * chunk_size]
                    future = pool.submit(cls._invert_dcs_for, chunk_kwargs, chunk_common_kwargs)
                    future.add_done_callback(lambda fut: pbar.update(len(fut.result())))
                    futures.append(future)
        return sum([future.result() for future in futures], [])

    
    @classmethod  # pylint: disable-next=too-many-arguments
    def from_dispersion_field(cls, dispersion_field, chunk_size=10, n_workers=None, bar=True, **kwargs):
        dc_list = cls._invert_dc_parallel(dispersion_field.items, [kwargs] * len(dispersion_field.items), chunk_size, n_workers, bar, desc="VS laws inverted")
        return cls(items=dc_list, auto_create_interpolator=True, is_geographic=dispersion_field.is_geographic)

