import warnings

import polars as pl

from ..utils import to_list


def get_uphole_correction_method(survey, uphole_correction_method):
    if uphole_correction_method not in {"auto", "time", "depth", None}:
        raise ValueError

    if uphole_correction_method == "auto":
        if not survey.is_uphole:
            return None
        return "time" if "SourceUpholeTime" in survey.available_headers else "depth"

    if uphole_correction_method == "time" and "SourceUpholeTime" not in survey.available_headers:
        raise ValueError
    if uphole_correction_method == "depth" and "SourceDepth" not in survey.available_headers:
        raise ValueError
    return uphole_correction_method


def group_source_headers(survey, index_cols):
    index_cols = to_list(index_cols)
    all_cols_set = set(index_cols + ["SourceX", "SourceY", "SourceSurfaceElevation"])
    if "SourceDepth" in survey.available_headers:
        all_cols_set.add("SourceDepth")
    if "SourceUpholeTime" in survey.available_headers:
        all_cols_set.add("SourceUpholeTime")
    all_cols = list(all_cols_set)
    non_index_cols = list(all_cols_set - set(index_cols))

    headers = pl.from_pandas(survey.get_headers(all_cols), rechunk=False)
    duplicated_expr = pl.any_horizontal([pl.n_unique(col) > 1 for col in non_index_cols]).alias("HasDuplicatedHeaders")
    headers = headers.groupby(index_cols).agg(pl.mean(non_index_cols), duplicated_expr).to_pandas()
    if headers["HasDuplicatedHeaders"].any():
        warnings.warn("Some sources have non-unique locations or uphole data. Statics may be inaccurate.")
    return headers


def group_receiver_headers(survey, index_cols):
    index_cols = to_list(index_cols)
    all_cols_set = set(index_cols + ["GroupX", "GroupY", "ReceiverGroupElevation"])
    all_cols = list(all_cols_set)
    non_index_cols = list(all_cols_set - set(index_cols))

    headers = pl.from_pandas(survey.get_headers(all_cols), rechunk=False)
    duplicated_expr = pl.any_horizontal([pl.n_unique(col) > 1 for col in non_index_cols]).alias("HasDuplicatedHeaders")
    headers = headers.groupby(index_cols).agg(pl.mean(non_index_cols), duplicated_expr).to_pandas()
    if headers["HasDuplicatedHeaders"].any():
        warnings.warn("Some receivers have non-unique locations. Statics may be inaccurate.")
    return headers
