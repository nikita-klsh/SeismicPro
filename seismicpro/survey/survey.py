from functools import partial

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm
from batchflow import Pipeline

from .loader import Loader, DummyLoader, SEGYLoader
from .batch import SeismicBatch
from .plot_geometry import SurveyGeometryPlot
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
                       sample_interval=None, delay=0, limits=None, validate=True, endian="big", chunk_size=25000,
                       n_workers=None, bar=True):
        loader = SEGYLoader(path, sample_interval=sample_interval, delay=delay, limits=limits, endian=endian)

        header_cols = header_cols  # TODO: merge with indexed_by, source_id_cols and receiver_id_cols

        pbar = partial(tqdm, desc="Trace headers loaded") if bar else False
        headers = loader.load_headers(header_cols, chunk_size=chunk_size, n_workers=n_workers, pbar=pbar)
        headers = SurveyTraceHeaders(headers, indexed_by=indexed_by, source_id_cols=source_id_cols,
                                     receiver_id_cols=receiver_id_cols, validate=validate)
        return cls(headers, loader)

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        return cls.from_segy_file(path, *args, **kwargs)

    def clone(self):
        return type(self)(self.headers.clone(), self.loader.clone())

    # Implement methods for trace headers processing

    def filter(self, cond, cols=None, axis=None, unpack_args=False, inplace=False, return_mask=False,
               preserve_geometry=True, **kwargs):
        if not inplace:
            self = self.clone()  # pylint: disable=self-cls-assignment
        _, mask = self.headers.filter(cond, cols, axis=axis, unpack_args=unpack_args, inplace=True, return_mask=True,
                                      preserve_geometry=preserve_geometry, **kwargs)
        if return_mask:
            return self, mask
        return self

    def apply(self, func, cols, res_cols=None, axis=None, unpack_args=False, inplace=False, **kwargs):
        if not inplace:
            self = self.clone()  # pylint: disable=self-cls-assignment
        self.headers.apply(func, cols, res_cols=res_cols, axis=axis, unpack_args=unpack_args, inplace=True, **kwargs)
        return self

    def reindex(self, index=None, inplace=False):
        if not inplace:
            self = self.clone()  # pylint: disable=self-cls-assignment
        self.headers.reindex(index, inplace=True)
        return self

    # Gather loading methods

    def load_gather(self, headers, limits=None, **loader_kwargs):
        data, sample_interval, delay = self.load_traces(headers, limits=limits, return_samples_info=True,
                                                        **loader_kwargs)
        return Gather(headers=headers, data=data, sample_interval=sample_interval, delay=delay, survey=self)

    def get_gather(self, index, limits=None, **loader_kwargs):
        return self.load_gather(self.get_headers_by_indices([index,]), limits=limits, **loader_kwargs)

    def get_gathers(self, indices, limits=None, **loader_kwargs):
        headers, n_traces = self.get_headers_by_indices(indices, return_n_traces=True)
        gather = self.load_gather(headers, limits=limits, **loader_kwargs)
        bounds = np.cumsum(np.concatenate([[0], n_traces]))
        gathers = [gather[start:stop] for start, stop in zip(bounds[:-1], bounds[1:])]
        return gathers

    def sample_gather(self, limits=None, **loader_kwargs):
        return self.get_gather(index=np.random.choice(self.indices), limits=limits, **loader_kwargs)

    # BatchFlow dataset interface

    @property
    def p(self):
        return self.pipeline()

    def pipeline(self, *args, **kwargs):
        return Pipeline(self, *args, **kwargs)

    def __rshift__(self, other):
        if not isinstance(other, Pipeline):
            raise TypeError(f"Pipeline is expected, but got {type(other)}. Use as survey >> pipeline")
        pipeline = other.from_pipeline(other)
        pipeline.dataset = self
        return pipeline

    def __lshift__(self, other):
        return self >> other

    def gen_batch(self, batch_size, shuffle=False, n_iters=None, n_epochs=None, drop_last=False,
                  notifier=False, iter_params=None, component=None):
        if self.index is None:
            raise ValueError
        for indices in self.index.gen_batch(batch_size, shuffle, n_iters, n_epochs, drop_last, notifier, iter_params):
            batch = self.create_batch(indices, component=component)
            yield batch

    def next_batch(self, batch_size, shuffle=False, n_iters=None, n_epochs=None, drop_last=False,
                   iter_params=None, component=None):
        batch_index = self.index.next_batch(batch_size, shuffle, n_iters, n_epochs, drop_last, iter_params)
        return self.create_batch(batch_index, component=component)

    def create_batch(self, index, component=None):
        if component is None:
            raise ValueError
        batch = SeismicBatch(index)
        batch.add_components(component, init=batch.array_of_nones)
        for i, gather in enumerate(self.get_gathers(self.indices[index.indices])):
            getattr(batch, component)[i] = gather
        return batch

    # Visualization

    def plot_geometry(self, **kwargs):
        SurveyGeometryPlot(self, **kwargs).plot()
