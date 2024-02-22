from functools import cached_property

import pandas as pd

from .trace_headers import TraceHeaders
from ..utils import to_list
from ..utils.coordinates import Coordinates, get_coords_cols


class GatherTraceHeaders(TraceHeaders):
    def __init__(self, headers, indexed_by=None, coords_cols=None):
        super().__init__(headers, indexed_by=indexed_by)

        if coords_cols is not None:
            # raise KeyError if non-coordinate columns are used or they don't exist and unify the order of coords
            coords_cols = self._validate_columns(get_coords_cols(coords_cols))
        else:
            # try inferring coords_cols by indexed_by if possible, don't raise exceptions in this case
            try:
                coords_cols = self._validate_columns(get_coords_cols(indexed_by))
            except KeyError:
                pass
        self.coords_cols = coords_cols

    @cached_property
    def index(self):
        if self.indexed_by is None:
            return None
        index_headers = self[self.indexed_by]
        if (index_headers == index_headers[0]).all():
            index = index_headers[0]
            if index.size == 1:
                return index.item()
            return tuple(index)
        return None

    def invalidate_index(self):
        self.__dict__.pop("index", None)

    @cached_property
    def coords(self):
        if self.coords_cols is None:
            return None
        coords = [int(coord) if coord.is_integer() else coord for coord in self[self.coords_cols].mean(axis=0)]
        return Coordinates(coords, self.coords_cols)

    def invalidate_coords(self):
        self.__dict__.pop("coords", None)

    def invalidate_cache(self, changed_cols=None):
        if changed_cols is None:
            self.invalidate_index()
            self.invalidate_coords()
            return

        changed_cols_set = set(to_list(changed_cols))
        if self.indexed_by is not None and changed_cols_set & set(to_list(self.indexed_by)):
            self.invalidate_index()
        if self.indexed_by is not None and changed_cols_set & set(to_list(self.coords_cols)):
            self.invalidate_coords()

    def clone_cached_properties(self, other):
        """Clone calculated cached properties to self from other."""
        if "index" in other.__dict__:
            self.__dict__["index"] = other.index
        if "coords" in other.__dict__:
            self.__dict__["coords"] = other.coords

    def clone(self):
        cloned = type(self)(pd.DataFrame(self.headers), indexed_by=self.indexed_by, coords_cols=self.coords_cols)
        cloned.clone_cached_properties(self)
        return cloned

    # Invalidate cache if headers have changed

    def __setitem__(self, key, value):
        """Set given values to selected headers."""
        super().__setitem__(key, value)
        self.invalidate_cache(key)

    def filter(self, cond, cols=None, axis=None, unpack_args=False, inplace=False, return_mask=False, **kwargs):
        old_n_traces = self.n_traces
        res, mask = super().filter(cond, cols=cols, axis=axis, unpack_args=unpack_args, inplace=inplace,
                                   return_mask=True, **kwargs)
        if res.n_traces < old_n_traces:
            self.invalidate_cache()  # Index and coords may change if they were not unique

        if return_mask:
            return res, mask
        return res

    def apply(self, func, cols, res_cols=None, axis=None, unpack_args=False, inplace=False, **kwargs):
        res = super().apply(func, cols, res_cols=res_cols, axis=axis, unpack_args=unpack_args, inplace=inplace,
                            **kwargs)
        self.invalidate_cache(cols if res_cols is None else res_cols)
        return res
