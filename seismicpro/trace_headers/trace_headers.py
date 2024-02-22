import warnings

import numpy as np
import pandas as pd
import polars as pl

from ..utils import to_list


class TraceHeaders:
    def __init__(self, headers, indexed_by=None):
        if isinstance(headers, pl.DataFrame):
            headers = headers.to_pandas(use_pyarrow_extension_array=True)
        elif not isinstance(headers, pd.DataFrame):
            raise TypeError
        self.headers = headers
        self.indexed_by = self._validate_columns(indexed_by)

    def __contains__(self, item):
        return item in self.headers

    def _repr_html_(self):
        with pd.option_context("display.max_columns", None):
            return self.headers._repr_html_()

    @property
    def n_traces(self):
        """int: The number of traces."""
        return len(self.headers)

    @property
    def is_empty(self):
        """bool: Whether no traces are stored in the container."""
        return self.n_traces == 0

    @property
    def headers_polars(self):
        """Return trace headers as a `polars.DataFrame`."""
        return pl.from_pandas(self.headers, rechunk=False, include_index=False)

    def __getitem__(self, key):
        """Select values of trace headers by their names and return them as an `np.ndarray`. The returned array will be
        1d if `key` is `str` and 2d if `key` is `list` of `str`."""
        return self.get_headers(key).to_numpy()

    def __setitem__(self, key, value):
        """Set given values to selected headers."""
        self.headers[key] = value

    def get_headers(self, cols):
        if not isinstance(cols, str):
            cols = to_list(cols)
        return self.headers[cols]

    def _validate_columns(self, columns=None):
        if columns is None:
            return None
        columns = tuple(to_list(columns))
        if not columns:
            raise KeyError
        if not all(col in self for col in columns):
            raise KeyError
        if len(columns) == 1:
            return columns[0]
        return columns

    def clone(self):
        return type(self)(pd.DataFrame(self.headers), indexed_by=self.indexed_by)

    @staticmethod
    def _apply(func, df, axis, unpack_args, **kwargs):
        """Apply a function to a `pd.DataFrame` along the specified axis."""
        if axis is None:
            args = (col_val for _, col_val in df.items()) if unpack_args else (df,)
            res = func(*args, **kwargs)
        else:
            # FIXME: Workaround for a pandas bug https://github.com/pandas-dev/pandas/issues/34822
            # raw=True causes incorrect apply behavior when axis=1 and multiple values are returned from `func`
            raw = axis != 1

            apply_func = (lambda args, **kwargs: func(*args, **kwargs)) if unpack_args else func
            res = df.apply(apply_func, axis=axis, raw=raw, result_type="expand", **kwargs)

        # Convert np.ndarray/pd.Series/pd.DataFrame outputs from `func` to a 2d array
        return pd.DataFrame(res).to_numpy()

    def filter(self, cond, cols=None, axis=None, unpack_args=False, inplace=False, return_mask=False, **kwargs):
        """Keep only those rows of `headers` where `cond` is `True`."""
        if not inplace:
            self = self.clone()  # pylint: disable=self-cls-assignment

        if cols is None:
            mask = cond
        else:
            headers = self.get_headers(to_list(cols))
            mask = self._apply(cond, headers, axis=axis, unpack_args=unpack_args, **kwargs)
            if (mask.ndim != 2) or (mask.shape[1] != 1):
                raise ValueError("cond must return a single value for each header row")
            mask = mask[:, 0]
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
            raise ValueError("cond must return a bool value for each header row")

        self.headers = self.headers.loc[mask]
        if self.is_empty:
            warnings.warn("Empty headers after filtering", RuntimeWarning)

        if return_mask:
            return self, mask
        return self

    def apply(self, func, cols, res_cols=None, axis=None, unpack_args=False, inplace=False, **kwargs):
        """Apply a function to `self.headers` along the specified axis."""
        if not inplace:
            self = self.clone()  # pylint: disable=self-cls-assignment

        cols = to_list(cols)
        res_cols = cols if res_cols is None else to_list(res_cols)
        res = self._apply(func, self.get_headers(cols), axis=axis, unpack_args=unpack_args, **kwargs)
        self[res_cols] = res
        return self
