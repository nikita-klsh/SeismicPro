import numpy as np
import pandas as pd
import polars as pl

from .loader import Loader, DummyLoader, SEGYLoader
from ..gather import Gather
from ..trace_headers import SurveyTraceHeaders
from ..containers import SamplesContainer
from ..decorators import delegate_calls


@delegate_calls(Loader, "loader")
@delegate_calls(SurveyTraceHeaders, "headers")
class Survey(SamplesContainer):
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

    def load_gather(self, headers, limits=None, **loader_kwargs):
        data, sample_interval, delay = self.loader.load_traces(headers, limits=limits, return_samples_info=True,
                                                               **loader_kwargs)
        return Gather(headers=headers, data=data, sample_interval=sample_interval, delay=delay, survey=self)

    def get_gather(self, index, limits=None, **loader_kwargs):
        return self.load_gather(self.headers.get_headers_by_indices([index,]), limits=limits, **loader_kwargs)

    def sample_gather(self, limits=None, **loader_kwargs):
        return self.get_gather(index=np.random.choice(self.headers.indices), limits=limits, **loader_kwargs)
