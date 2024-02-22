from textwrap import wrap

import numpy as np
import polars as pl
from sklearn.neighbors import RadiusNeighborsRegressor

from .indexer import Indexer
from ..utils import to_list


def format_warning(warning, warning_list=None, width=80):
    """Format a list of warnings into a single string."""
    warning_msg = "\n".join(wrap(warning, width=width))
    n_warnings = len(warning_list)
    if n_warnings > 0:
        ix_len = len(str(n_warnings))
        wrap_space = 3 + ix_len
        wrap_sep = "\n" + " " * wrap_space
        warning_list = [wrap_sep.join(wrap(warn_str, width=width-wrap_space)) for warn_str in warning_list]
        warning_msg += "".join([f"\n\n {i+1:{ix_len}d}. {warn_str}" for i, warn_str in enumerate(warning_list)])

    warning_sep = "-" * width
    return "".join(["\n\n", warning_sep, "\n\n", warning_msg, "\n\n", warning_sep])


def isclose_polars(expr1, expr2, rtol=1e-5, atol=1e-8):
    """Return whether two `polars` expressions are close within given tolerances."""
    return (expr1 - expr2).abs() <= atol + rtol * expr2.abs()


# pylint: disable-next=too-many-statements
def validate_trace_headers(headers, offset_atol=10, cdp_atol=10, elevation_atol=5, elevation_radius=50, warn_width=80):
    """Validate trace headers for consistency and return either a string with found problems or `None` if all checks
    have successfully passed or no checks can been performed. `headers` is expected to be a `polars.DataFrame`."""
    if headers.is_empty():
        return None

    n_traces = len(headers)
    loaded_columns = set(headers.columns)
    empty_mask = headers.select((pl.col("*") == 0).all()).row(0, named=True)
    empty_columns = {col for col, mask in empty_mask.items() if mask}
    non_empty_columns = loaded_columns - empty_columns

    shot_coords_cols = ["SourceX", "SourceY"]
    rec_coords_cols = ["GroupX", "GroupY"]
    cdp_coords_cols = ["CDP_X", "CDP_Y"]

    msg_list = []
    if empty_columns:
        msg_list.append("Empty headers: " + ", ".join(empty_columns))

    expr_list = []
    if {"FieldRecord", "TraceNumber"} <= non_empty_columns:
        sorted_ids = pl.struct("FieldRecord", "TraceNumber").sort()
        n_uniques = (sorted_ids != sorted_ids.shift(1)).sum() + 1  # Much faster than direct n_unique call for structs
        expr_list.append((n_traces - n_uniques).alias("n_duplicated"))

    if "SourceUpholeTime" in non_empty_columns:
        expr_list.append((pl.col(["SourceUpholeTime"]) < 0).sum().alias("n_neg_uphole_time"))

    if "SourceDepth" in non_empty_columns:
        expr_list.append((pl.col(["SourceDepth"]) < 0).sum().alias("n_neg_uphole_depth"))

    if {"SourceUpholeTime", "SourceDepth"} <= loaded_columns:
        zero_time = isclose_polars(pl.col("SourceUpholeTime"), pl.lit(0))
        zero_depth = isclose_polars(pl.col("SourceDepth"), pl.lit(0))
        n_zero_time = pl.when(zero_time & ~zero_depth).then(1).otherwise(0).sum().alias("n_zero_time")
        n_zero_depth = pl.when(~zero_time & zero_depth).then(1).otherwise(0).sum().alias("n_zero_depth")
        expr_list.extend([n_zero_time, n_zero_depth])

    if "offset" in non_empty_columns:
        expr_list.append((pl.col(["offset"]) < 0).sum().alias("n_neg_offsets"))

    if {*shot_coords_cols, *rec_coords_cols, "offset"} <= non_empty_columns:
        calculated_offsets = ((pl.col("SourceX") - pl.col("GroupX"))**2 +
                              (pl.col("SourceY") - pl.col("GroupY"))**2).sqrt()
        n_close_offsets = isclose_polars(calculated_offsets, pl.col("offset"), rtol=0, atol=offset_atol).sum()
        n_not_close_offsets = (n_traces - n_close_offsets).alias("n_not_close_offsets")
        expr_list.append(n_not_close_offsets)

    if {*shot_coords_cols, *rec_coords_cols, *cdp_coords_cols} <= non_empty_columns:
        calculated_cdp_x = (pl.col("SourceX") + pl.col("GroupX")) / 2
        calculated_cdp_y = (pl.col("SourceY") + pl.col("GroupY")) / 2
        dist = ((calculated_cdp_x - pl.col("CDP_X"))**2 + (calculated_cdp_y - pl.col("CDP_Y"))**2).sqrt()
        n_not_close_cdp = (n_traces - isclose_polars(dist, pl.lit(0), rtol=0, atol=10).sum()).alias("n_not_close_cdp")
        expr_list.append(n_not_close_cdp)

    if expr_list:
        expr_res = headers.select(expr_list).row(0, named=True)

        n_duplicated = expr_res.get("n_duplicated", 0)
        if n_duplicated:
            msg_list.append(f"Non-unique traces identifier (FieldRecord, TraceNumber) for {n_duplicated} traces "
                            f"({(n_duplicated / n_traces):.2%})")

        n_neg_uphole_time = expr_res.get("n_neg_uphole_time", 0)
        if n_neg_uphole_time:
            msg_list.append(f"Negative uphole times for {n_neg_uphole_time} traces "
                            f"({(n_neg_uphole_time / n_traces):.2%})")

        n_neg_uphole_depth = expr_res.get("n_neg_uphole_depth", 0)
        if n_neg_uphole_depth:
            msg_list.append(f"Negative uphole depths for {n_neg_uphole_depth} traces "
                            f"({(n_neg_uphole_depth / n_traces):.2%})")

        n_zero_time = expr_res.get("n_zero_time", 0)
        if n_zero_time:
            msg_list.append(f"Zero uphole time for non-zero uphole depth for {n_zero_time} traces "
                            f"({(n_zero_time / n_traces):.2%})")

        n_zero_depth = expr_res.get("n_zero_depth", 0)
        if n_zero_depth:
            msg_list.append(f"Zero uphole depth for non-zero uphole time for {n_zero_depth} traces "
                            f"({(n_zero_depth / n_traces):.2%})")

        n_neg_offsets = expr_res.get("n_neg_offsets", 0)
        if n_neg_offsets:
            msg_list.append(f"Negative offsets for {n_neg_offsets} traces ({(n_neg_offsets / n_traces):.2%})")

        n_not_close_offsets = expr_res.get("n_not_close_offsets", 0)
        if n_not_close_offsets:
            msg_list.append("Distance between source (SourceX, SourceY) and receiver (GroupX, GroupY) differs from "
                            f"the corresponding offset by more than {offset_atol} meters for {n_not_close_offsets} "
                            f"traces ({(n_not_close_offsets / n_traces):.2%})")

        n_not_close_cdp = expr_res.get("n_not_close_cdp", 0)
        if n_not_close_cdp:
            msg_list.append("A midpoint between source (SourceX, SourceY) and receiver (GroupX, GroupY) differs from "
                            f"the corresponding coordinates (CDP_X, CDP_Y) by more than {cdp_atol} meters for "
                            f"{n_not_close_cdp} traces ({(n_not_close_cdp / n_traces):.2%})")

    if {*shot_coords_cols, "SourceSurfaceElevation"} <= non_empty_columns:
        shot_elevations = headers.select(*shot_coords_cols, "SourceSurfaceElevation").unique(maintain_order=False)
        is_duplicated_expr = (pl.col("SourceSurfaceElevation").count() > 1).alias("duplicated")
        is_duplicated = shot_elevations.groupby(shot_coords_cols).agg(is_duplicated_expr)
        n_duplicated = is_duplicated.select(pl.col("duplicated").sum()).item()
        if n_duplicated:
            msg_list.append(f"Non-unique surface elevation (SourceSurfaceElevation) for {n_duplicated} source "
                            f"locations ({(n_duplicated / len(is_duplicated)):.2%})")

    if {*rec_coords_cols, "ReceiverGroupElevation"} <= non_empty_columns:
        rec_elevations = headers.select(*rec_coords_cols, "ReceiverGroupElevation").unique(maintain_order=False)
        is_duplicated_expr = (pl.col("ReceiverGroupElevation").count() > 1).alias("duplicated")
        is_duplicated = rec_elevations.groupby(rec_coords_cols).agg(is_duplicated_expr)
        n_duplicated = is_duplicated.select(pl.col("duplicated").sum()).item()
        if n_duplicated:
            msg_list.append(f"Non-unique surface elevation (ReceiverGroupElevation) for {n_duplicated} receiver "
                            f"locations ({(n_duplicated / len(is_duplicated)):.2%})")

    if {*shot_coords_cols, *rec_coords_cols, "ReceiverGroupElevation", "SourceSurfaceElevation"} <= non_empty_columns:
        elevations = np.concatenate([shot_elevations.to_numpy(), rec_elevations.to_numpy()]).astype(np.float32)
        rnr = RadiusNeighborsRegressor(radius=elevation_radius).fit(elevations[:, :2], elevations[:, 2])
        close_mask = np.isclose(rnr.predict(elevations[:, :2]), elevations[:, 2], rtol=0, atol=elevation_atol)
        n_diff = (~close_mask).sum()
        if n_diff:
            msg_list.append("Surface elevations of sources (SourceSurfaceElevation) and receivers "
                            f"(ReceiverGroupElevation) differ by more than {elevation_atol} meters within spatial "
                            f"radius of {elevation_radius} meters for {n_diff} sensor locations "
                            f"({(n_diff / len(elevations)):.2%})")

    return format_warning("The survey has the following inconsistencies in trace headers:", msg_list, width=warn_width)


def calculate_source_headers(headers, source_id_cols=None, validate=True, warn_width=80):
    """Validate source-related trace headers for consistency and return either a string with found problems or
    `None` if all checks have successfully passed or no checks can been performed. `headers` is expected to be a
    `polars.DataFrame`."""
    if headers.is_empty() or source_id_cols is None:
        return None, None, None

    source_id_cols = to_list(source_id_cols)
    empty_id_mask = headers.select((pl.col(source_id_cols) == 0).all()).row(0, named=True)
    empty_id_cols = [col for col, mask in empty_id_mask.items() if mask]
    if empty_id_cols:
        if validate:
            warn_str = ("No checks of source-related trace headers were performed since the following source ID "
                        f"headers are empty: {', '.join(empty_id_cols)}")
            return None, None, format_warning(warn_str, width=warn_width)
        return None, None, None

    # Manually iterate over SOURCE_HEADERS to preserve order
    source_cols = ("SourceX", "SourceY", "SourceSurfaceElevation", "SourceUpholeTime", "SourceDepth")
    source_non_id_cols = [col for col in source_cols if col in headers.columns and col not in source_id_cols]
    expr_list = [pl.col("locs"), pl.count("locs").alias("n_rows"), pl.mean(source_non_id_cols)]
    if validate:
        expr_list.append(pl.n_unique(source_non_id_cols).name.prefix("n_"))
    grouped = headers.with_row_index("locs").group_by(source_id_cols).agg(expr_list).sort(source_id_cols)
    source_headers = grouped.select(source_id_cols + source_non_id_cols)
    indexer = grouped.select(source_id_cols + ["locs", "n_rows"]).to_pandas(use_pyarrow_extension_array=True)
    indexer.set_index(source_id_cols, inplace=True)
    indexer = Indexer(indexer)
    if not validate:
        return source_headers, indexer, None

    n_duplicated = grouped.select([pl.len().alias("n_sources"), (pl.col("^n_.*$") > 1).sum()])
    n_duplicated = n_duplicated.row(0, named=True)
    n_sources = n_duplicated.pop("n_sources")

    msg_list = []

    n_duplicated_x = n_duplicated.get("n_SourceX", 0)
    if n_duplicated_x:
        msg_list.append(f"Non-unique X source coordinate (SourceX) for {n_duplicated_x} sources "
                        f"({(n_duplicated_x / n_sources):.2%})")

    n_duplicated_y = n_duplicated.get("n_SourceY", 0)
    if n_duplicated_y:
        msg_list.append(f"Non-unique Y source coordinate (SourceY) for {n_duplicated_y} sources "
                        f"({(n_duplicated_y / n_sources):.2%})")

    n_duplicated_elevations = n_duplicated.get("n_SourceSurfaceElevation", 0)
    if n_duplicated_elevations:
        msg_list.append(f"Non-unique surface elevation (SourceSurfaceElevation) for {n_duplicated_elevations} sources "
                        f"({(n_duplicated_elevations / n_sources):.2%})")

    n_duplicated_uphole_times = n_duplicated.get("n_SourceUpholeTime", 0)
    if n_duplicated_uphole_times:
        msg_list.append(f"Non-unique source uphole time (SourceUpholeTime) for {n_duplicated_uphole_times} sources "
                        f"({(n_duplicated_uphole_times / n_sources):.2%})")

    n_duplicated_depths = n_duplicated.get("n_SourceDepth", 0)
    if n_duplicated_depths:
        msg_list.append(f"Non-unique source depth (SourceDepth) for {n_duplicated_depths} sources "
                        f"({(n_duplicated_depths / n_sources):.2%})")

    if msg_list:
        warn_str = "Selected source ID columns result in the following inconsistencies of trace headers:"
        warn_str = format_warning(warn_str, msg_list, width=warn_width)
        return source_headers, indexer, warn_str
    return source_headers, indexer, None


def calculate_receiver_headers(headers, receiver_id_cols=None, validate=True, warn_width=80):
    """Validate receiver-related trace headers for consistency and return either a string with found problems or `None`
    if all checks have successfully passed or no checks can been performed. `headers` is expected to be a
    `polars.DataFrame`."""
    if headers.is_empty() or receiver_id_cols is None:
        return None, None, None

    receiver_id_cols = to_list(receiver_id_cols)
    empty_id_mask = headers.select((pl.col(receiver_id_cols) == 0).all()).row(0, named=True)
    empty_id_cols = [col for col, mask in empty_id_mask.items() if mask]
    if empty_id_cols:
        if validate:
            warn_str = ("No checks of receiver-related trace headers were performed since the following receiver "
                        f"ID headers are empty: {', '.join(empty_id_cols)}")
            return None, None, format_warning(warn_str, width=warn_width)
        return None, None, None

    # Manually iterate over RECEIVER_HEADERS to preserve order
    receiver_cols = ("GroupX", "GroupY", "ReceiverGroupElevation")
    receiver_non_id_cols = [col for col in receiver_cols if col in headers.columns and col not in receiver_id_cols]
    expr_list = [pl.col("locs"), pl.count("locs").alias("n_rows"), pl.mean(receiver_non_id_cols)]
    if validate:
        expr_list.append(pl.n_unique(receiver_non_id_cols).name.prefix("n_"))
    grouped = headers.with_row_index("locs").group_by(receiver_id_cols).agg(expr_list).sort(receiver_id_cols)
    receiver_headers = grouped.select(receiver_id_cols + receiver_non_id_cols)
    indexer = grouped.select(receiver_id_cols + ["locs", "n_rows"]).to_pandas(use_pyarrow_extension_array=True)
    indexer.set_index(receiver_id_cols, inplace=True)
    indexer = Indexer(indexer)
    if not validate:
        return receiver_headers, indexer, None

    n_duplicated = grouped.select([pl.len().alias("n_receivers"), (pl.col("^n_.*$") > 1).sum()])
    n_duplicated = n_duplicated.row(0, named=True)
    n_receivers = n_duplicated.pop("n_receivers")

    msg_list = []

    n_duplicated_x = n_duplicated.get("n_GroupX", 0)
    if n_duplicated_x:
        msg_list.append(f"Non-unique X receiver coordinate (GroupX) for {n_duplicated_x} receivers "
                        f"({(n_duplicated_x / n_receivers):.2%})")

    n_duplicated_y = n_duplicated.get("n_GroupY", 0)
    if n_duplicated_y:
        msg_list.append(f"Non-unique Y receiver coordinate (GroupY) for {n_duplicated_y} receivers "
                        f"({(n_duplicated_y / n_receivers):.2%})")

    n_duplicated_elevations = n_duplicated.get("n_ReceiverGroupElevation", 0)
    if n_duplicated_elevations:
        msg_list.append(f"Non-unique surface elevation (ReceiverGroupElevation) for {n_duplicated_elevations} "
                        f"receivers ({(n_duplicated_elevations / n_receivers):.2%})")

    if msg_list:
        warn_str = "Selected receiver ID columns result in the following inconsistencies of trace headers:"
        warn_str = format_warning(warn_str, msg_list, width=warn_width)
        return receiver_headers, indexer, warn_str
    return receiver_headers, indexer, None


def calculate_bin_headers(headers, validate=True, warn_width=80):
    if headers.is_empty():
        return None, None, None, None

    if ({"INLINE_3D", "CROSSLINE_3D"} <= set(headers.columns) and
        any(headers.select((pl.col("INLINE_3D", "CROSSLINE_3D") != 0).any()).row(0))):
        bin_id_cols = ("INLINE_3D", "CROSSLINE_3D")
    elif "CDP" in headers.columns and headers.select((pl.col("CDP") != 0).any()).item():
        bin_id_cols = "CDP"
    else:
        if validate:
            warn_str = "Either none of CDP, INLINE_3D or CROSSLINE_3D headers were loaded or they are empty"
            return None, None, None, format_warning(warn_str, width=warn_width)
        return None, None, None, None
    bin_id_cols_list = to_list(bin_id_cols)

    if ({"CDP_X", "CDP_Y"} <= set(headers.columns) and
        any(headers.select((pl.col("CDP_X", "CDP_Y") == 0).all()).row(0))):
        bin_indexer = Indexer.from_dataframe(headers, bin_id_cols)
        bin_headers = bin_indexer.indexer.index.to_frame(index=False)
        if validate:
            warn_str = "At least one of CDP_X, CDP_Y coordinates is not loaded or empty"
            return bin_headers, bin_indexer, None, format_warning(warn_str, width=warn_width)
        return bin_headers, bin_indexer, None, None

    headers = headers.lazy().with_row_index("locs")
    common_expr_list = [pl.col("locs"), pl.count("locs").alias("n_rows")]
    bin_expr_list = [pl.mean("CDP_X", "CDP_Y"), pl.n_unique("CDP_X", "CDP_Y").name.prefix("n_")]
    cdp_expr_list = [pl.mean(bin_id_cols), pl.n_unique(bin_id_cols).name.prefix("n_")]
    grouped_bin = headers.group_by(bin_id_cols).agg(common_expr_list + bin_expr_list).sort(bin_id_cols)
    grouped_cdp = headers.group_by("CDP_X", "CDP_Y").agg(common_expr_list + cdp_expr_list).sort("CDP_X", "CDP_Y")
    grouped_bin, grouped_cdp = pl.collect_all([grouped_bin, grouped_cdp])

    bin_headers = grouped_bin.select(bin_id_cols_list + ["CDP_X", "CDP_Y"])
    bin_indexer = grouped_bin.select(bin_id_cols_list + ["locs", "n_rows"]).to_pandas(use_pyarrow_extension_array=True)
    bin_indexer.set_index(bin_id_cols_list, inplace=True)
    bin_indexer = Indexer(bin_indexer)
    cdp_indexer = grouped_cdp.select("CDP_X", "CDP_Y", "locs", "n_rows").to_pandas(use_pyarrow_extension_array=True)
    cdp_indexer.set_index(["CDP_X", "CDP_Y"], inplace=True)
    cdp_indexer = Indexer(cdp_indexer)
    if not validate:
        return bin_headers, bin_indexer, cdp_indexer, None

    msg_list = []

    n_duplicated_cdps = grouped_bin.select(((pl.col("n_CDP_X") > 1) | (pl.col("n_CDP_Y") > 1)).sum()).item()
    if n_duplicated_cdps:
        msg_list.append(f"Non-unique midpoint coordinates (CDP_X, CDP_Y) for {n_duplicated_cdps} bins "
                        f"({(n_duplicated_cdps / len(grouped_bin)):.2%})")

    if bin_id_cols == "CDP":
        duplicated_expr = (pl.col("n_CDP") > 1).sum()
    else:
        duplicated_expr = ((pl.col("n_INLINE_3D") > 1) | (pl.col("n_CROSSLINE_3D") > 1)).sum()
    n_duplicated_bins = grouped_cdp.select(duplicated_expr).item()
    if n_duplicated_bins:
        msg_list.append(f"Non-unique bin for {n_duplicated_bins} midpoint locations "
                        f"({(n_duplicated_bins / len(grouped_cdp)):.2%})")

    if msg_list:
        warn_str = "The following inconsistencies of trace headers defining geometry bins are found:"
        warn_str = format_warning(warn_str, msg_list, width=warn_width)
        return bin_headers, bin_indexer, cdp_indexer, warn_str
    return bin_headers, bin_indexer, cdp_indexer, None
