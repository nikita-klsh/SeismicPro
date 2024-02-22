"""Core classes and functions of SeismicPro"""

# Enable copy-on-write
import pandas as pd
pd.set_option("mode.copy_on_write", True)

from .config import config
from .dataset import SeismicDataset
from .index import SeismicIndex
from .batch import SeismicBatch
from .survey import Survey
from .gather import Gather, CroppedGather, SignalLeakage
from .velocity_spectrum import VerticalVelocitySpectrum, ResidualVelocitySpectrum, SlantStack
from .muter import Muter, MuterField
from .stacking_velocity import StackingVelocity, StackingVelocityField
from .refractor_velocity import RefractorVelocity, RefractorVelocityField
from .metrics import Metric, PipelineMetric, MetricMap
from .utils import aggregate_segys, make_prestack_segy


__version__ = "2.1.0"
