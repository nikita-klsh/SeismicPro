import warnings
from inspect import getmembers
from functools import cached_property

import numpy as np
import pandas as pd

from .trace_headers import TraceHeaders
from .indexer import Indexer
from .geometry import infer_geometry
from .validation import (validate_trace_headers, calculate_source_headers, calculate_receiver_headers,
                         calculate_bin_headers)
from ..utils import to_list
from ..utils.interpolation import IDWInterpolator, DelaunayInterpolator, CloughTocherInterpolator, RBFInterpolator


class SurveyTraceHeaders(TraceHeaders):
    PUBLIC_ATTRIBUTES = ["indexers", "indexed_by"]
    PUBLIC_PROPERTIES = ["n_traces", "is_empty", "indexer", "indices", "n_gathers", "index", "source_id_cols",
                         "source_headers", "is_uphole", "n_sources", "receiver_id_cols", "receiver_headers",
                         "n_receivers", "bin_headers", "bin_id_cols", "n_bins", "is_stacked", "is_2d", "is_3d",
                         "elevation_interpolator", "geometry", "cached_properties", "calculated_cached_properties",
                         "cached_properties_cols"]
    PUBLIC_METHODS = ["__getitem__", "__setitem__", "get_headers", "add_indexer", "create_indexer", "reset",
                      "set_source_id_cols", "calculate_source_headers", "set_receiver_id_cols",
                      "calculate_receiver_headers", "calculate_bin_headers", "validate_headers",
                      "get_elevation_interpolator", "create_elevation_interpolator",
                      "create_default_elevation_interpolator", "infer_geometry", "invalidate_indexers",
                      "invalidate_cache", "get_traces_locs", "get_headers_by_indices"]

    def __init__(self, headers, indexed_by=None, source_id_cols=None, receiver_id_cols=None, indexers=None,
                 validate=True, infer_geometry=True):
        super().__init__(headers, indexed_by=indexed_by)

        if indexers is None:
            indexers = {}
        if not isinstance(indexers, dict):
            raise TypeError
        self.indexers = indexers

        if source_id_cols is None:
            if "FieldRecord" in self:
                source_id_cols = "FieldRecord"
            elif "SourceX" in self and "SourceY" in self:
                source_id_cols = ("SourceX", "SourceY")
        self._source_id_cols = self._validate_columns(source_id_cols)

        if receiver_id_cols is None and "GroupX" in self and "GroupY" in self:
            receiver_id_cols = ("GroupX", "GroupY")
        self._receiver_id_cols = self._validate_columns(receiver_id_cols)

        if validate:
            self.validate_headers()
        if infer_geometry:
            self.create_default_elevation_interpolator()
            self.infer_geometry()

        # Reindex after validate and infer_geometry: an appropriate indexer may have been already created
        self.reindex(indexed_by, inplace=True)

    # Indexer properties and processing methods

    @property
    def indexer(self):
        if self.indexed_by is None:
            return None
        return self.indexers[self.indexed_by]

    @property
    def indices(self):
        if self.indexer is None:
            return None
        return self.indexer.indices

    @property
    def n_gathers(self):
        if self.indices is None:
            return None
        return len(self.indices)

    @property
    def index(self):
        if self.indexer is None:
            return None
        return self.indexer.index

    def add_indexer(self, indexer=None):
        if indexer is None:
            return
        if not isinstance(indexer, Indexer):
            raise TypeError
        self.indexers[indexer.index_cols] = indexer

    def create_indexer(self, indexed_by):
        indexed_by = self._validate_columns(indexed_by)
        if indexed_by not in self.indexers:
            indexer = Indexer.from_dataframe(self.headers_polars, indexed_by)
            self.add_indexer(indexer)

    def reset(self, what="iter"):
        if self.indexer is not None:
            self.indexer.reset(what=what)

    # Source headers properties and processing methods

    @property
    def source_id_cols(self):
        return self._source_id_cols

    @source_id_cols.setter
    def source_id_cols(self, cols):
        self.set_source_id_cols(cols)

    @cached_property
    def source_headers(self):
        self.calculate_source_headers(validate=True)
        return self.source_headers

    @cached_property
    def is_uphole(self):
        """bool or None: Whether the survey is uphole. `None` if uphole-related headers are not loaded."""
        has_uphole_times = "SourceUpholeTime" in self.source_headers
        has_uphole_depths = "SourceDepth" in self.source_headers
        has_positive_uphole_times = has_uphole_times and (self.source_headers["SourceUpholeTime"] > 0).any()
        has_positive_uphole_depths = has_uphole_depths and (self.source_headers["SourceDepth"] > 0).any()
        if not has_uphole_times and not has_uphole_depths:
            return None
        return has_positive_uphole_times or has_positive_uphole_depths

    @property
    def n_sources(self):
        if self.source_headers is None:
            return None
        return len(self.source_headers)

    def set_source_id_cols(self, cols, validate=True):
        """Set new trace headers that uniquely identify a seismic source and optionally validate consistency of
        source-related trace headers by checking that each source has unique coordinates, surface elevation, uphole
        time and depth."""
        cols = self._validate_columns(cols)
        if self.source_id_cols != cols:
            self.__dict__.pop("source_headers", None)
            self.__dict__.pop("is_uphole", None)
        self._source_id_cols = cols
        if validate:
            self.calculate_source_headers(validate=validate)

    def _calculate_source_headers(self, validate=True):
        res = calculate_source_headers(self.headers_polars, self.source_id_cols, validate=validate)
        source_headers, source_indexer, warn_str = res
        self.__dict__["source_headers"] = source_headers
        self.add_indexer(source_indexer)
        return warn_str

    def calculate_source_headers(self, validate=True):
        warn_str = self._calculate_source_headers(validate=validate)
        if warn_str is not None:
            warnings.warn(warn_str, RuntimeWarning)

    # Receiver headers properties and processing methods

    @property
    def receiver_id_cols(self):
        return self._receiver_id_cols

    @receiver_id_cols.setter
    def receiver_id_cols(self, cols):
        self.set_receiver_id_cols(cols)

    @cached_property
    def receiver_headers(self):
        self.calculate_receiver_headers(validate=True)
        return self.receiver_headers

    @property
    def n_receivers(self):
        if self.receiver_headers is None:
            return None
        return len(self.receiver_headers)

    def set_receiver_id_cols(self, cols, validate=True):
        """Set new trace headers that uniquely identify a receiver and optionally validate consistency of
        receiver-related trace headers by checking that each receiver has unique coordinates and surface elevation."""
        cols = self._validate_columns(cols)
        if self.receiver_id_cols != cols:
            self.__dict__.pop("receiver_headers", None)
        self._receiver_id_cols = cols
        if validate:
            self.calculate_receiver_headers(validate=validate)

    def _calculate_receiver_headers(self, validate=True):
        res = calculate_receiver_headers(self.headers_polars, self.receiver_id_cols, validate=validate)
        receiver_headers, receiver_indexer, warn_str = res
        self.__dict__["receiver_headers"] = receiver_headers
        self.add_indexer(receiver_indexer)
        return warn_str

    def calculate_receiver_headers(self, validate=True):
        warn_str = self._calculate_receiver_headers(validate=validate)
        if warn_str is not None:
            warnings.warn(warn_str, RuntimeWarning)

    # Bin headers properties and processing methods

    @cached_property
    def bin_headers(self):
        self.calculate_bin_headers(validate=True)
        return self.bin_headers

    @cached_property
    def bin_id_cols(self):
        self.calculate_bin_headers(validate=True)
        return self.bin_id_cols

    @property
    def n_bins(self):
        if self.bin_headers is None:
            return None
        return len(self.bin_headers)

    @property
    def is_stacked(self):
        if self.n_bins is None:
            return None
        return self.n_traces == self.n_bins

    @property
    def is_2d(self):
        if self.bin_id_cols is None:
            return None
        return self.bin_id_cols == "CDP"

    @property
    def is_3d(self):
        if self.bin_id_cols is None:
            return None
        return self.bin_id_cols == ("INLINE_3D", "CROSSLINE_3D")

    def _calculate_bin_headers(self, validate=True):
        res = calculate_bin_headers(self.headers_polars, validate=validate)
        bin_headers, bin_indexer, cdp_indexer, warn_str = res
        bin_id_cols = None if bin_indexer is None else bin_indexer.index_cols
        self.__dict__["bin_headers"] = bin_headers
        self.__dict__["bin_id_cols"] = bin_id_cols
        self.add_indexer(bin_indexer)
        self.add_indexer(cdp_indexer)
        return warn_str

    def calculate_bin_headers(self, validate=True):
        warn_str = self._calculate_bin_headers(validate=validate)
        if warn_str is not None:
            warnings.warn(warn_str, RuntimeWarning)

    # Headers validation

    def validate_headers(self, offset_atol=10, cdp_atol=10, elevation_atol=5, elevation_radius=50):
        warn_list = [
            validate_trace_headers(self.headers_polars, offset_atol=offset_atol, cdp_atol=cdp_atol,
                                   elevation_atol=elevation_atol, elevation_radius=elevation_radius),
            self._calculate_source_headers(validate=True),
            self._calculate_receiver_headers(validate=True),
            self._calculate_bin_headers(validate=True),
        ]
        warn_list = [warn for warn in warn_list if warn is not None]
        if warn_list:
            warn_list = [warn.rstrip("-\n") for warn in warn_list[:-1]] + [warn_list[-1]]
            warn_list = [warn_list[0]] + [warn.lstrip("-\n") for warn in warn_list[1:]]
            warnings.warn("\n\n\n".join(warn_list), RuntimeWarning)

    # Elevation interpolator calculation

    @property
    def _available_elevation_interpolators(self):
        """dict: A mapping from names of available elevation interpolators to the corresponding classes."""
        interpolators = {
            "idw": IDWInterpolator,
            "delaunay": DelaunayInterpolator,
            "ct": CloughTocherInterpolator,
            "rbf": RBFInterpolator,
        }
        return interpolators

    def _get_elevation_interpolator_class(self, interpolator):
        """Chooses appropriate interpolator type by its name defined by `interpolator` and a mapping returned by
        `self._available_elevation_interpolators`."""
        interpolator_class = self._available_elevation_interpolators.get(interpolator)
        if interpolator_class is None:
            raise ValueError(f"Unknown interpolator {interpolator}. Available options are: "
                             f"{', '.join(self._available_elevation_interpolators.keys())}")
        return interpolator_class

    def _get_elevation_data(self, use_source_headers=None, use_receiver_headers=None):
        source_elevation_headers = ["SourceX", "SourceY", "SourceSurfaceElevation"]
        receiver_elevation_headers = ["GroupX", "GroupY", "ReceiverGroupElevation"]

        can_use_source_headers = (self.source_headers is not None and
                                  set(source_elevation_headers) <= set(self.source_headers.columns))
        can_use_receiver_headers = (self.receiver_headers is not None and
                                    set(receiver_elevation_headers) <= set(self.receiver_headers.columns))

        if use_source_headers is None:
            use_source_headers = can_use_source_headers
        if use_receiver_headers is None:
            use_receiver_headers = can_use_receiver_headers
        if not use_source_headers and not use_receiver_headers:
            raise ValueError("Either both source and receiver headers are not selected or their elevation-related "
                             "headers are not loaded")

        elevation_data_list = []
        if use_source_headers:
            if not can_use_source_headers:
                raise ValueError("Elevation-related headers of seismic sources are not loaded")
            elevation_data_list.append(self.source_headers[source_elevation_headers].to_numpy())
        if use_receiver_headers:
            if not can_use_receiver_headers:
                raise ValueError("Elevation-related headers of seismic receivers are not loaded")
            elevation_data_list.append(self.receiver_headers[receiver_elevation_headers].to_numpy())
        return np.concatenate(elevation_data_list), use_source_headers, use_receiver_headers

    def get_elevation_interpolator(self, interpolator, *, use_source_headers=None, use_receiver_headers=None,
                                   **kwargs):
        res = self._get_elevation_data(use_source_headers, use_receiver_headers)
        elevation_data, uses_source_headers, uses_receiver_headers = res
        interpolator_class = self._get_elevation_interpolator_class(interpolator)
        interpolator = interpolator_class(elevation_data[:, :2], elevation_data[:, 2], **kwargs)
        interpolator.uses_source_headers = uses_source_headers
        interpolator.uses_receiver_headers = uses_receiver_headers
        return interpolator

    def create_elevation_interpolator(self, interpolator, use_source_headers=None, use_receiver_headers=None,
                                      **kwargs):
        interpolator = self.get_elevation_interpolator(interpolator, use_source_headers=use_source_headers,
                                                       use_receiver_headers=use_receiver_headers, **kwargs)
        self.__dict__["elevation_interpolator"] = interpolator

    def create_default_elevation_interpolator(self):
        try:
            self.create_elevation_interpolator("idw", neighbors=4)
        except ValueError:
            self.__dict__["elevation_interpolator"] = None

    @cached_property
    def elevation_interpolator(self):
        self.create_default_elevation_interpolator()
        return self.elevation_interpolator

    # Geometry calculation

    @cached_property
    def geometry(self):
        self.infer_geometry()
        return self.geometry

    def infer_geometry(self):
        if self.bin_headers is None:
            geometry = None
        else:
            geometry = infer_geometry(self.bin_headers)
        self.__dict__["geometry"] = geometry

    # Cache invalidation

    @property
    def cached_properties(self):
        return [prop for prop, _ in getmembers(type(self), lambda x: isinstance(x, cached_property))]

    @property
    def calculated_cached_properties(self):
        return [prop for prop in self.cached_properties if prop in self.__dict__]

    @property
    def cached_properties_cols(self):
        cols_dict = {}

        if self.source_id_cols is not None:
            source_id_cols_set = set(to_list(self.source_id_cols))
            cols_dict["source_headers"] = source_id_cols_set | {"SourceX", "SourceY", "SourceSurfaceElevation",
                                                                "SourceUpholeTime", "SourceDepth"}
            cols_dict["is_uphole"] = source_id_cols_set | {"SourceUpholeTime", "SourceDepth"}

        if self.receiver_id_cols is not None:
            receiver_id_cols_set = set(to_list(self.receiver_id_cols))
            cols_dict["receiver_headers"] = receiver_id_cols_set | {"GroupX", "GroupY", "ReceiverGroupElevation"}

        elevation_interpolator = self.__dict__.get("elevation_interpolator")
        if elevation_interpolator is not None:
            elevation_interpolator_cols = {}
            if elevation_interpolator.uses_source_headers:
                elevation_interpolator_cols |= {"SourceX", "SourceY", "SourceSurfaceElevation"}
            if elevation_interpolator.uses_receiver_headers:
                elevation_interpolator_cols |= {"GroupX", "GroupY", "ReceiverGroupElevation"}
            cols_dict["elevation_interpolator"] = elevation_interpolator_cols

        cols_dict["bin_headers"] = {"CDP", "INLINE_3D", "CROSSLINE_3D", "CDP_X", "CDP_Y"}
        cols_dict["bin_id_cols"] = {"CDP", "INLINE_3D", "CROSSLINE_3D"}
        cols_dict["geometry"] = {"CDP", "INLINE_3D", "CROSSLINE_3D", "CDP_X", "CDP_Y"}
        return cols_dict

    def invalidate_indexers(self, updated_cols=None, changed_n_rows=False):
        # Create a new indexers dict so that other surveys are not affected
        if changed_n_rows:
            self.indexers = {}
        elif updated_cols is not None:
            updated_cols_set = set(to_list(updated_cols))
            self.indexers = {indexed_by: indexer for indexed_by, indexer in self.indexers.items()
                            if not set(to_list(indexed_by)) & updated_cols_set}

        if self.indexed_by not in self.indexers:
            self.create_indexer(self.indexed_by)

    def invalidate_cache(self, updated_cols=None, changed_n_rows=False, preserve_geometry=False):
        self.invalidate_indexers(updated_cols, changed_n_rows)

        if changed_n_rows:
            calculated_cached_properties = set(self.calculated_cached_properties)
            if preserve_geometry:
                calculated_cached_properties -= {"geometry", "elevation_interpolator"}
            for prop in calculated_cached_properties:
                self.__dict__.pop(prop)
        elif updated_cols is not None:
            updated_cols_set = set(to_list(updated_cols))
            for prop, prop_cols in self.cached_properties_cols.items():
                if updated_cols_set & prop_cols:
                    self.__dict__.pop(prop, None)

    # Invalidate cache if headers have changed

    def __setitem__(self, key, value):
        """Set given values to selected headers."""
        super().__setitem__(key, value)
        self.invalidate_cache(key)

    def filter(self, cond, cols=None, axis=None, unpack_args=False, inplace=False, return_mask=False,
               preserve_geometry=True, **kwargs):
        old_n_traces = self.n_traces
        res, mask = super().filter(cond, cols=cols, axis=axis, unpack_args=unpack_args, inplace=inplace,
                                   return_mask=True, **kwargs)
        if res.n_traces < old_n_traces:
            self.invalidate_cache(changed_n_rows=True, preserve_geometry=preserve_geometry)

        if return_mask:
            return res, mask
        return res

    def apply(self, func, cols, res_cols=None, axis=None, unpack_args=False, inplace=False, **kwargs):
        res = super().apply(func, cols, res_cols=res_cols, axis=axis, unpack_args=unpack_args, inplace=inplace,
                            **kwargs)
        self.invalidate_cache(cols if res_cols is None else res_cols)
        return res

    # Clone survey trace headers and its cache

    def clone_cached_properties(self, other):
        """Clone calculated cached properties to self from other."""
        for prop in other.calculated_cached_properties:
            val = getattr(other, prop)
            if isinstance(val, pd.DataFrame):
                val = pd.DataFrame(val)
            self.__dict__[prop] = val

    def clone(self):
        cloned = type(self)(pd.DataFrame(self.headers), indexed_by=self.indexed_by, source_id_cols=self.source_id_cols,
                            receiver_id_cols=self.receiver_id_cols, indexers=self.indexers, validate=False)
        cloned.clone_cached_properties(self)
        return cloned

    # Headers processing methods

    def get_traces_locs(self, indices, return_n_traces=False):
        return self.indexer.get_locs(indices, return_n_rows=return_n_traces)

    def get_headers_by_indices(self, indices, return_n_traces=False):
        locs, n_traces = self.get_traces_locs(indices, return_n_traces=True)
        headers = self.headers.iloc[locs]
        if return_n_traces:
            return headers, n_traces
        return headers

    def reindex(self, indexed_by=None, inplace=False):
        if not inplace:
            self = self.clone()  # pylint: disable=self-cls-assignment

        if indexed_by is not None:
            indexed_by = self._validate_columns(indexed_by)
            self.create_indexer(indexed_by)
        self.indexed_by = indexed_by
        self.reset()
        return self
