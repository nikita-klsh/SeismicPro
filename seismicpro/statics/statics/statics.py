import warnings
from functools import partial

import numpy as np
import pandas as pd
import polars as pl

from .statics_plot import StaticsPlot
from ..utils import group_source_headers, group_receiver_headers
from ...survey import Survey
from ...metrics import MetricMap
from ...utils import to_list, align_args


class Statics:
    def __init__(self, survey, source_statics, receiver_statics, source_id_cols=None, source_statics_col="Statics",
                 source_surface_statics_col=None, receiver_id_cols=None, receiver_statics_col="Statics"):
        self.is_single_survey = isinstance(survey, Survey)
        self.survey_list = to_list(survey)

        # Set source statics and validate them for consistency
        self.source_statics_list = None
        self.source_id_cols = None
        self.source_statics_col = None
        self.source_surface_statics_col = None
        self.source_elevation_map = None
        self.source_statics_map = None
        self.source_surface_statics_map = None
        self._set_source_statics(source_statics, source_id_cols, source_statics_col, source_surface_statics_col)

        # Set receiver statics and validate them for consistency
        self.receiver_statics_list = None
        self.receiver_id_cols = None
        self.receiver_statics_col = None
        self.receiver_elevation_map = None
        self.receiver_statics_map = None
        self._set_receiver_statics(receiver_statics, receiver_id_cols, receiver_statics_col)

    @property
    def n_surveys(self):
        return len(self.survey_list)

    def _set_source_statics(self, source_statics, source_id_cols=None, source_statics_col="Statics",
                            source_surface_statics_col=None):
        source_statics_list = to_list(source_statics)
        if len(source_statics_list) not in {1, self.n_surveys}:
            raise ValueError

        # Set source_surface_statics_col to default if not given but stored in the statics df
        statics_cols_list = [source_statics_col]
        if source_surface_statics_col is None and all("SurfaceStatics" in statics for statics in source_statics_list):
            source_surface_statics_col = "SurfaceStatics"
            statics_cols_list.append(source_surface_statics_col)

        # Infer source_id_cols if not given
        if source_id_cols is None:
            source_id_cols = set(source_statics_list[0].columns) - set(statics_cols_list)
            if any(set(statics.columns) - set(statics_cols_list) != source_id_cols for statics in source_statics_list):
                raise ValueError
        source_id_cols = to_list(source_id_cols)

        # Check whether statics are unique for each source
        source_statics_list_pl = []
        for statics in source_statics_list:
            statics_pl = pl.from_pandas(statics, rechunk=False)
            if len(statics_pl.select(source_id_cols).unique()) != len(statics):
                warnings.warn("Source statics contain sources with duplicated indices. "
                              "Corresponding statics are averaged.")
                statics_pl = statics_pl.groupby(source_id_cols).agg(pl.mean(statics_cols_list))
            source_statics_list_pl.append(statics_pl)
        if len(source_statics_list_pl) == 1:
            source_statics_list_pl = source_statics_list_pl * self.n_surveys

        # Extract source-related headers from each survey and check whether surface statics can be reconstructed
        # if not given
        source_headers_list_pl = [pl.from_pandas(group_source_headers(survey, source_id_cols), rechunk=False)
                                  for survey in self.survey_list]
        has_uphole_time = all("SourceUpholeTime" in headers for headers in source_headers_list_pl)
        reconstruct_surface_statics = source_surface_statics_col is None and has_uphole_time
        if reconstruct_surface_statics:
            source_surface_statics_col = "SurfaceStatics"
            statics_cols_list.append(source_surface_statics_col)

        # Merge statics with source-related headers, reconstruct surface statics if needed and check whether statics
        # exist for all sources
        res_source_statics_list = []
        for source_headers_pl, statics_pl in zip(source_headers_list_pl, source_statics_list_pl):
            statics_pl = source_headers_pl.join(statics_pl, on=source_id_cols, how="left")
            if reconstruct_surface_statics:
                reconstruct_expr = pl.col(source_statics_col) + pl.col("SourceUpholeTime")
                statics_pl = statics_pl.with_columns(reconstruct_expr.alias(source_surface_statics_col))
            if any((statics_pl.select(source_statics_col).null_count() > 0).row(0)):
                warnings.warn("Source statics miss some sources from the survey. Their statics will be set to 0")
                statics_pl = statics_pl.with_columns(pl.col(statics_cols_list).fill_null(0))
            res_source_statics_list.append(statics_pl.to_pandas())

        # Construct statics maps
        map_statics, map_id_cols = self._concatenate_statics(res_source_statics_list, source_id_cols)
        map_cls = partial(MetricMap, coords=map_statics[["SourceX", "SourceY"]], index=map_statics[map_id_cols])
        source_elevation_map = map_cls(values=map_statics["SourceSurfaceElevation"])
        source_statics_map = map_cls(values=map_statics[source_statics_col])
        if source_surface_statics_col is None:
            source_surface_statics_map = None
        else:
            source_surface_statics_map = map_cls(values=map_statics[source_surface_statics_col])

        # Set computed attributes
        self.source_statics_list = res_source_statics_list
        self.source_id_cols = source_id_cols
        self.source_statics_col = source_statics_col
        self.source_surface_statics_col = source_surface_statics_col
        self.source_elevation_map = source_elevation_map
        self.source_statics_map = source_statics_map
        self.source_surface_statics_map = source_surface_statics_map

    def _set_receiver_statics(self, receiver_statics, receiver_id_cols=None, receiver_statics_col="Statics"):
        receiver_statics_list = to_list(receiver_statics)
        if len(receiver_statics_list) not in {1, self.n_surveys}:
            raise ValueError

        # Infer receiver_id_cols if not given
        if receiver_id_cols is None:
            receiver_id_cols = set(receiver_statics_list[0].columns) - set(receiver_statics_col)
            if any(set(statics.columns) - set(receiver_statics_col) != receiver_id_cols
                   for statics in receiver_statics_list):
                raise ValueError
        receiver_id_cols = to_list(receiver_id_cols)

        # Check whether statics are unique for each receiver
        receiver_statics_list_pl = []
        for statics in receiver_statics_list:
            statics_pl = pl.from_pandas(statics, rechunk=False)
            if len(statics_pl.select(receiver_id_cols).unique()) != len(statics):
                warnings.warn("Receiver statics contain receivers with duplicated indices. "
                              "Corresponding statics are averaged.")
                statics_pl = statics_pl.groupby(receiver_id_cols).agg(pl.mean(receiver_statics_col))
            receiver_statics_list_pl.append(statics_pl)
        if len(receiver_statics_list_pl) == 1:
            receiver_statics_list_pl = receiver_statics_list_pl * self.n_surveys

        # Extract receiver-related headers from each survey
        receiver_headers_list_pl = [pl.from_pandas(group_receiver_headers(survey, receiver_id_cols), rechunk=False)
                                    for survey in self.survey_list]

        # Merge statics with receiver-related headers and check whether statics exist for all sources
        res_receiver_statics_list = []
        for receiver_headers_pl, statics_pl in zip(receiver_headers_list_pl, receiver_statics_list_pl):
            statics_pl = receiver_headers_pl.join(statics_pl, on=receiver_id_cols, how="left")
            if statics_pl.select(receiver_statics_col).null_count().item():
                warnings.warn("Receiver statics miss some receivers from the survey. Their statics will be set to 0")
                statics_pl = statics_pl.with_columns(pl.col(receiver_statics_col).fill_null(0))
            res_receiver_statics_list.append(statics_pl.to_pandas())

        # Construct statics maps
        map_statics, map_id_cols = self._concatenate_statics(res_receiver_statics_list, receiver_id_cols)
        map_cls = partial(MetricMap, coords=map_statics[["GroupX", "GroupY"]], index=map_statics[map_id_cols])
        receiver_elevation_map = map_cls(values=map_statics["ReceiverGroupElevation"])
        receiver_statics_map = map_cls(values=map_statics[receiver_statics_col])

        # Set computed attributes
        self.receiver_statics_list = res_receiver_statics_list
        self.receiver_id_cols = receiver_id_cols
        self.receiver_statics_col = receiver_statics_col
        self.receiver_elevation_map = receiver_elevation_map
        self.receiver_statics_map = receiver_statics_map

    @staticmethod
    def _concatenate_statics(statics_list, id_cols):
        if len(statics_list) == 1:
            return statics_list[0], id_cols
        statics = pd.concat(statics_list)
        statics["Part"] = np.concatenate([np.full(len(stat), i) for i, stat in enumerate(statics_list)])
        id_cols = ["Part"] + to_list(id_cols)
        return statics, id_cols

    # Statics application

    def _apply_to_container(self, container, source_statics, receiver_statics, statics_header="Statics"):
        container_headers = container.get_polars_headers()
        loaded_headers = container_headers.columns
        indexed_by = container.indexed_by

        source_statics = pl.from_pandas(source_statics, rechunk=False)
        source_statics_expr = pl.col(self.source_statics_col).alias("_SourceStatics")
        source_statics = source_statics.select(*self.source_id_cols, source_statics_expr)
        container_headers = container_headers.join(source_statics, how="left", on=self.source_id_cols)

        receiver_statics = pl.from_pandas(receiver_statics, rechunk=False)
        receiver_statics_expr = pl.col(self.receiver_statics_col).alias("_ReceiverStatics")
        receiver_statics = receiver_statics.select(*self.receiver_id_cols, receiver_statics_expr)
        container_headers = container_headers.join(receiver_statics, how="left", on=self.receiver_id_cols)

        statics_expr = (pl.col("_SourceStatics") + pl.col("_ReceiverStatics")).alias(statics_header)
        headers = container_headers.select(*loaded_headers, statics_expr).to_pandas()
        headers.set_index(indexed_by, inplace=True)

        statics_container = container.copy(ignore="headers")
        statics_container.headers = headers
        return statics_container

    def apply(self, statics_header="Statics"):
        _, statics_header = align_args(self.survey_list, statics_header)
        data_iterator = zip(self.survey_list, self.source_statics_list, self.receiver_statics_list, statics_header)
        statics_survey_list = [self._apply_to_container(survey, source_statics, receiver_statics, header)
                               for survey, source_statics, receiver_statics, header in data_iterator]
        if self.is_single_survey:
            return statics_survey_list[0]
        return statics_survey_list

    # Statics visualization

    def plot(self, by, center=True, sort_by=None, gather_plot_kwargs=None, **kwargs):
        statics_plot = StaticsPlot(self, by=by, center=center, sort_by=sort_by, gather_plot_kwargs=gather_plot_kwargs,
                                   **kwargs)
        statics_plot.plot()
