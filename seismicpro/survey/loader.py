import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from segfast import MemmapLoader

from ..containers import SamplesContainer
from ..utils import get_first_defined, ForPoolExecutor


class Loader(SamplesContainer):
    def __init__(self, *args, n_samples, sample_interval, delay=0, limits=None, **kwargs):
        _ = args, kwargs

        if n_samples <= 0:
            raise ValueError
        if sample_interval <= 0:
            raise ValueError("Sample interval must be positive, please provide a valid sample_interval")
        self.file_samples = self.create_samples(n_samples, sample_interval, delay)
        self.file_sample_interval = sample_interval
        self.file_delay = delay

        # Set samples and sample_rate according to passed `limits`
        self.limits = None
        self.samples = None
        self.sample_interval = None
        self.delay = None
        self.set_limits(limits)

    @property
    def file_sample_rate(self):
        """float: Sample rate of seismic traces in the source SEG-Y file. Measured in Hz."""
        return 1000 / self.file_sample_interval

    @property
    def file_n_samples(self):
        """int: Trace length in samples in the source SEG-Y file."""
        return len(self.file_samples)
    
    def _process_limits(self, limits=None):
        """Convert given `limits` to a `slice`."""
        if limits is None:
            return slice(0, self.file_n_samples, 1)
        if isinstance(limits, int):
            limits = slice(limits)
        elif isinstance(limits, (tuple, list)):
            limits = slice(*limits)

        # Use .indices to avoid negative slicing range
        indices = limits.indices(self.file_n_samples)
        if indices[-1] < 0:
            raise ValueError("Negative step is not allowed.")
        if indices[1] <= indices[0]:
            raise ValueError("Empty traces after setting limits.")
        return slice(*indices)

    def _get_limits_info(self, limits):
        """Convert given `limits` to a `slice` and return it together with the number of samples, sample interval and
        delay recording time these limits imply."""
        limits = self._process_limits(limits)
        samples = self.file_samples[limits]
        return limits, len(samples), self.file_sample_interval * limits.step, samples[0]

    def set_limits(self, limits):
        """Update default survey time limits that are used during trace loading and statistics calculation.

        Parameters
        ----------
        limits : int or tuple or slice
            Default time limits to be used during trace loading and survey statistics calculation. `int` or `tuple` are
            used as arguments to init a `slice`. The resulting object is stored in `self.limits` attribute and used to
            recalculate `self.samples`, `self.sample_interval` and `self.delay`. Measured in samples.

        Raises
        ------
        ValueError
            If negative step of limits was passed.
            If the resulting samples length is zero.
        """
        self.limits, _, self.sample_interval, self.delay = self._get_limits_info(limits)
        self.samples = self.file_samples[self.limits]

    def clone(self):
        raise NotImplementedError

    def load_traces(self, headers, limits=None, buffer=None, return_samples_info=False):
        _ = headers, limits, buffer, return_samples_info
        raise NotImplementedError
    

class DummyLoader(Loader):
    def __init__(self, *, n_samples=500, sample_interval=2, delay=0, limits=None):
        super().__init__(n_samples=n_samples, sample_interval=sample_interval, delay=delay, limits=limits)

    def clone(self):
        return type(self)(n_samples=self.file_n_samples, sample_interval=self.file_sample_interval,
                          delay=self.file_delay, limits=self.limits)

    def load_traces(self, headers, limits=None, buffer=None, return_samples_info=False):
        limits, n_samples, sample_interval, delay = self._get_limits_info(get_first_defined(limits, self.limits))
        if buffer is None:
            buffer = np.empty((len(headers), n_samples), dtype=np.float32)
        buffer[:] = 0

        if return_samples_info:
            return buffer, sample_interval, delay
        return buffer


class SEGYLoader(Loader):
    def __init__(self, path, *, sample_interval=None, delay=0, limits=None, endian="big"):
        self.path = path
        self.loader = MemmapLoader(path, endian=endian)
        sample_interval = get_first_defined(sample_interval, self.loader.sample_interval / 1000)
        super().__init__(n_samples=self.loader.n_samples, sample_interval=sample_interval, delay=delay, limits=limits)

    def clone(self):
        return type(self)(path=self.path, sample_interval=self.file_sample_interval, delay=self.file_delay,
                          limits=self.limits, endian=self.loader.endian)

    def load_traces(self, headers, limits=None, buffer=None, chunk_size=None, n_workers=None,
                    return_samples_info=False):
        indices = headers["TRACE_SEQUENCE_FILE"].to_numpy() - 1

        if chunk_size is None:
            chunk_size = len(indices)
        n_chunks, last_chunk_size = divmod(len(indices), chunk_size)
        chunk_sizes = [chunk_size] * n_chunks
        if last_chunk_size:
            n_chunks += 1
            chunk_sizes += [last_chunk_size]
        chunk_borders = np.cumsum([0] + chunk_sizes)

        if n_workers is None:
            n_workers = os.cpu_count()
        n_workers = min(n_chunks, n_workers)
        executor_class = ForPoolExecutor if n_workers == 1 else ThreadPoolExecutor

        limits, n_samples, sample_interval, delay = self._get_limits_info(get_first_defined(limits, self.limits))
        if buffer is None:
            buffer = np.empty((len(indices), n_samples), dtype=self.loader.dtype)

        with executor_class(max_workers=n_workers) as pool:
            for start, end in zip(chunk_borders[:-1], chunk_borders[1:]):
                pool.submit(self.loader.load_traces, indices[start:end], limits=limits, buffer=buffer[start:end])

        if return_samples_info:
            return buffer, sample_interval, delay
        return buffer

    def load_headers(self, headers, *args, **kwargs):
        return self.loader.load_headers(headers, *args, **kwargs)
