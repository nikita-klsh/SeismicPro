import pandas as pd
import polars as pl

from .loader import DummyLoader, SEGYLoader
from ..trace_headers import SurveyTraceHeaders


class Survey:
    def __init__(self, headers, loader=None):
        if isinstance(headers, (pd.DataFrame, pl.DataFrame)):
            headers = SurveyTraceHeaders(headers)
        self.headers = headers

        if loader is None:
            loader = DummyLoader()
        self.loader = loader

    @classmethod
    def from_segy_file(cls, path, header_cols, indexed_by=None, source_id_cols=None, receiver_id_cols=None,
                       sample_interval=None, delay=0, limits=None, validate=True, infer_geometry=True, endian="big",
                       chunk_size=25000, n_workers=None, bar=True):
        loader = SEGYLoader(path, sample_interval=sample_interval, delay=delay, limits=limits, endian=endian)

        header_cols = header_cols  # TODO: merge with indexed_by, source_id_cols and receiver_id_cols
        headers = loader.load_headers(header_cols, chunk_size=chunk_size, n_workers=n_workers, bar=bar)
        headers = SurveyTraceHeaders(headers, indexed_by=indexed_by, source_id_cols=source_id_cols,
                                     receiver_id_cols=receiver_id_cols, validate=validate,
                                     infer_geometry=infer_geometry)
        return cls(headers, loader)

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        return cls.from_segy_file(path, *args, **kwargs)

    def clone(self):
        return type(self)(self.headers.clone(), self.loader.clone())
