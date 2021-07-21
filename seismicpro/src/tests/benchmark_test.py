"""Implementation of tests for benchmarks"""

import pytest

from seismicpro.benchmark import Benchmark
from seismicpro.src import Survey, SeismicDataset
from seismicpro.batchflow import Pipeline


#TODO:
# 1. Add tests that check corectness of benchmark.
# 2. Add check for save_to.
@pytest.mark.parametrize('method_name,method_kwargs,root_pipeline',
                         (('load', dict(src='raw'), None),
                          ('sort', dict(src='raw', by='offset'), Pipeline().load(src='raw', fmt='sgy'))))
def test_that_benchmark_runs_completely(segy_path, method_name, method_kwargs, root_pipeline):
    """Test benchmark"""
    survey = Survey(segy_path, header_index=['INLINE_3D', 'CROSSLINE_3D'], header_cols='offset', name='raw')
    dataset = SeismicDataset(surveys=survey)
    load_bm = Benchmark(method_name=method_name, method_kwargs=method_kwargs, targets=('for', 'threads'),
                        batch_sizes=[1, 5, 10], dataset=dataset, n_iters=3, root_pipeline=root_pipeline,
                        benchmark_cpu=True, save_to=None)
    load_bm.run()
