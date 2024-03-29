from functools import partial
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor


import h5py
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm
from batchflow import Pipeline, DatasetIndex

from .loader import Loader, DummyLoader, SEGYLoader, HDF5Loader, NPZLoader
from .batch import SeismicBatch
from .plot_geometry import SurveyGeometryPlot
from ..gather import Gather
from ..trace_headers import SurveyTraceHeaders
from ..containers import SamplesContainer
from ..decorators import delegate_calls


@delegate_calls(Loader, "loader")
@delegate_calls(SurveyTraceHeaders, "headers")
class Survey(SamplesContainer):
    def __init__(self, headers, loader=None, name=None):
        if isinstance(headers, (pd.DataFrame, pl.DataFrame)):
            headers = SurveyTraceHeaders(headers)
        self.headers = headers

        if loader is None:
            loader = DummyLoader()
        self.loader = loader

        self._name = None
        if name is not None:
            self.name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self.set_name(name)

    @property
    def has_name(self):
        return self.name is not None

    def set_name(self, name):
        if not isinstance(name, str):
            raise TypeError
        if not name.isidentifier():
            raise ValueError
        self._name = name

    @classmethod
    def from_segy_file(cls, path, header_cols, indexed_by=None, source_id_cols=None, receiver_id_cols=None,
                       sample_interval=None, delay=0, limits=None, validate=True, name=None, endian="big",
                       chunk_size=25000, n_workers=None, bar=True):
        loader = SEGYLoader(path, sample_interval=sample_interval, delay=delay, limits=limits, endian=endian)

        header_cols = header_cols  # TODO: merge with indexed_by, source_id_cols and receiver_id_cols

        pbar = partial(tqdm, desc="Trace headers loaded") if bar else False
        headers = loader.load_headers(header_cols, chunk_size=chunk_size, n_workers=n_workers, pbar=pbar)
        headers = SurveyTraceHeaders(headers, indexed_by=indexed_by, source_id_cols=source_id_cols,
                                     receiver_id_cols=receiver_id_cols, validate=validate)
        return cls(headers, loader, name=name)

    def to_hdf5(self, path, chunk_size=1000):
        file = h5py.File(path, 'w')
        file.attrs['sample_interval'] = self.sample_interval
        file.attrs['delay'] = self.delay
        buffer = file.create_dataset('data', shape=(self.n_traces, self.n_samples), dtype=self.loader.loader.dtype)
        self.load_traces(self.headers.headers, chunk_size=chunk_size, buffer=buffer)
        file.create_dataset('headers', data=self.headers.headers.to_records(index=False))
        file.close()
        return self
    
    @classmethod
    def from_hdf5(cls, path, header_cols, indexed_by=None, source_id_cols=None, receiver_id_cols=None,
                  limits=None, validate=True, name=None):
        file = h5py.File(path, 'r')
        loader = HDF5Loader(file, sample_interval=file.attrs['sample_interval'], delay=file.attrs['delay'], limits=limits)
        headers = pd.DataFrame.from_records(file['headers'][:])
        headers = SurveyTraceHeaders(headers, indexed_by=indexed_by, source_id_cols=source_id_cols,
                                     receiver_id_cols=receiver_id_cols, validate=validate)
        file.close()
        return cls(headers, loader, name=name)

    def to_npz(self, path, chunk_size=1000):
        import zipfile
        np.savez(path, headers=self.headers.headers.to_records(), sample_interval=self.sample_interval, delay=self.delay)
        with zipfile.ZipFile(path, 'a') as zf:
            with zf.open('data.npy', 'w', force_zip64=True) as fp:
                    header = {"shape": (self.n_traces, self.n_samples), "fortran_order": False, "descr": "f4"}
                    np.lib.format._write_array_header(fp, header)

                    indices = self["TRACE_SEQUENCE_FILE"] - 1

                    n_chunks, last_chunk_size = divmod(len(indices), chunk_size)
                    chunk_sizes = [chunk_size] * n_chunks
                    if last_chunk_size:
                        n_chunks += 1
                        chunk_sizes += [last_chunk_size]
                    chunk_borders = np.cumsum([0] + chunk_sizes)

                    for start, end in tqdm(zip(chunk_borders[:-1], chunk_borders[1:])):
                        traces = self.loader.loader.load_traces(indices[start:end])
                        fp.write(traces.tobytes('C'))

        return self


    @classmethod
    def from_npz(cls, path, header_cols, indexed_by=None, source_id_cols=None, receiver_id_cols=None,
                  limits=None, validate=True, name=None):
        file = np.load(path)
        loader = NPZLoader(file.zip, sample_interval=file['sample_interval'].item(), delay=file['delay'].item(), limits=limits)
        headers = pd.DataFrame.from_records(file['headers'][:])
        headers = SurveyTraceHeaders(headers, indexed_by=indexed_by, source_id_cols=source_id_cols,
                                     receiver_id_cols=receiver_id_cols, validate=validate)
        file.close()
        return cls(headers, loader, name=name)                            

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        return cls.from_segy_file(path, *args, **kwargs)

    def clone(self):
        return type(self)(self.headers.clone(), self.loader.clone())

    # Implement methods for trace headers processing

    def filter(self, cond, cols=None, axis=None, unpack_args=False, inplace=False, return_mask=False, **kwargs):
        if not inplace:
            self = self.clone()  # pylint: disable=self-cls-assignment
        _, mask = self.headers.filter(cond, cols, axis=axis, unpack_args=unpack_args, inplace=True, return_mask=True,
                                      **kwargs)
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
        return self.load_gather(self.get_headers_by_gather_indices([index,]), limits=limits, **loader_kwargs)

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

    def _gen_batch_sequential(self, batch_size, shuffle=False, n_iters=None, n_epochs=None, drop_last=False,
                              notifier=False, iter_params=None, component=None):
        for pos in self.index.gen_batch(batch_size, shuffle, n_iters, n_epochs, drop_last, notifier, iter_params):
            yield self.create_batch(pos, component=component)

    def _gen_batch_parallel(self, batch_size, shuffle=False, n_iters=None, n_epochs=None, drop_last=False,
                            notifier=False, iter_params=None, component=None, load_prefetch=None):
        LAST_BATCH_SIGNAL = -1
        batch_queue = Queue(load_prefetch)
        batch_creator = ThreadPoolExecutor(max_workers=load_prefetch, thread_name_prefix="Batch-loader")

        def create_batch():
            pos_gen = self.index.gen_batch(batch_size, shuffle, n_iters, n_epochs, drop_last, notifier, iter_params)
            for pos in pos_gen:
                future = batch_creator.submit(self.create_batch, pos, component=component)
                batch_queue.put(future, block=True)
            batch_queue.put(LAST_BATCH_SIGNAL, block=True)
        service_thread = Thread(target=create_batch, name="Batch-loader-service")
        service_thread.start()

        while True:
            future = batch_queue.get(block=True)
            if future == LAST_BATCH_SIGNAL:
                break
            yield future.result()

        batch_creator.shutdown()
        service_thread.join()

    def gen_batch(self, batch_size, shuffle=False, n_iters=None, n_epochs=None, drop_last=False, notifier=False,
                  iter_params=None, component=None, load_prefetch=4):
        if self.index is None:
            raise ValueError
        kwargs = {"batch_size": batch_size, "shuffle": shuffle, "n_iters": n_iters, "n_epochs": n_epochs,
                  "drop_last": drop_last, "notifier": notifier, "iter_params": iter_params, "component": component}
        if load_prefetch:
            return self._gen_batch_parallel(load_prefetch=load_prefetch, **kwargs)
        return self._gen_batch_sequential(**kwargs)

    def next_batch(self, batch_size, shuffle=False, n_iters=None, n_epochs=None, drop_last=False,
                   iter_params=None, component=None):
        if self.index is None:
            raise ValueError
        pos = self.index.next_batch(batch_size, shuffle, n_iters, n_epochs, drop_last, iter_params)
        return self.create_batch(pos, component=component)

    def _get_gather_component(self, positions, limits=None, **loader_kwargs):
        headers, n_traces = self.get_headers_by_gather_positions(positions, return_n_traces=True)
        gather = self.load_gather(headers, limits=limits, **loader_kwargs)
        bounds = np.cumsum(np.concatenate([[0], n_traces]))
        return np.array([gather[start:stop] for start, stop in zip(bounds[:-1], bounds[1:])])

    def create_batch(self, positions, component=None):
        if component is None:
            if not self.has_name:
                raise ValueError
            component = self.name
        if not isinstance(positions, DatasetIndex):
            positions = DatasetIndex(positions)
        batch = SeismicBatch(positions)
        batch.add_components(component, init=self._get_gather_component(positions.indices))
        return batch

    # Visualization

    def plot_geometry(self, **kwargs):
        SurveyGeometryPlot(self, **kwargs).plot()
