import math

import numpy as np
import polars as pl

from .dataset import TomoModelTravelTimeDataset
from ..grid import Grid
from ...utils import to_list, IDWInterpolator


class Grid3D(Grid):
    dataset_class = TomoModelTravelTimeDataset

    def __init__(self, origin, shape, cell_size, survey=None):
        super().__init__(survey)

        origin = np.broadcast_to(origin, 3).astype(np.float64)
        cell_size = np.broadcast_to(cell_size, 3).astype(np.float64)
        if (cell_size <= 0).any():
            raise ValueError
        shape = np.broadcast_to(shape, 3).astype(np.int32, casting="same_kind")
        if (shape <= 0).any():
            raise ValueError

        self.origin = origin
        self.cell_size = cell_size
        self.shape = shape

        self.z_cell_bounds = self.z_origin + self.z_cell_size * np.arange(self.n_z_cells + 1)
        self.x_cell_bounds = self.x_origin + self.x_cell_size * np.arange(self.n_x_cells + 1)
        self.y_cell_bounds = self.y_origin + self.y_cell_size * np.arange(self.n_y_cells + 1)

        self.z_cell_centers = self.z_cell_size / 2 + self.z_cell_bounds[:-1]
        self.x_cell_centers = self.x_cell_size / 2 + self.x_cell_bounds[:-1]
        self.y_cell_centers = self.y_cell_size / 2 + self.y_cell_bounds[:-1]

        self.air_mask = None
        if self.has_survey:
            # TODO: add surface elevation interpolator
            self._init_air_mask()

    @property
    def z_origin(self):
        return self.origin[0]

    @property
    def x_origin(self):
        return self.origin[1]

    @property
    def y_origin(self):
        return self.origin[2]

    @property
    def z_cell_size(self):
        return self.cell_size[0]

    @property
    def x_cell_size(self):
        return self.cell_size[1]

    @property
    def y_cell_size(self):
        return self.cell_size[2]

    @property
    def n_z_cells(self):
        return self.shape[0]

    @property
    def n_x_cells(self):
        return self.shape[1]

    @property
    def n_y_cells(self):
        return self.shape[2]

    @property
    def n_cells(self):
        return self.n_z_cells * self.n_x_cells * self.n_y_cells

    def _init_air_mask(self):
        headers = pl.concat([sur.get_polars_headers().lazy() for sur in to_list(self.survey)], rechunk=False)
        source_elevations = headers.select([
            pl.col("SourceX").alias("X"),
            pl.col("SourceY").alias("Y"),
            pl.col("SourceSurfaceElevation").alias("Elevation")
        ])
        receiver_elevations = headers.select([
            pl.col("GroupX").alias("X"),
            pl.col("GroupY").alias("Y"),
            pl.col("ReceiverGroupElevation").alias("Elevation")
        ])
        elevations = pl.concat([source_elevations, receiver_elevations], rechunk=False).select([
            ((pl.col("X") - self.x_origin) / self.x_cell_size).floor().cast(pl.Int32).alias("BinX"),
            ((pl.col("Y") - self.y_origin) / self.y_cell_size).floor().cast(pl.Int32).alias("BinY"),
            pl.col("Elevation")
        ]).groupby("BinX", "BinY").agg(pl.max("Elevation"))
        max_elevations = elevations.collect().to_numpy()

        used_cell_coords = self.origin[1:] + self.cell_size[1:] / 2 + self.cell_size[1:] * max_elevations[:, :2]
        max_elevation_interp = IDWInterpolator(used_cell_coords, max_elevations[:, 2], neighbors=8)

        cell_coords = np.array(np.meshgrid(self.x_cell_centers, self.y_cell_centers)).T.reshape(-1, 2)
        max_elevation_grid = max_elevation_interp(cell_coords).reshape(self.n_x_cells, self.n_y_cells)
        self.air_mask = self.z_cell_bounds[:-1, None, None] >= max_elevation_grid

    # IO

    @classmethod
    def from_survey(cls, survey, z_min, cell_size, spatial_margin=3):
        survey_list = to_list(survey)
        dz, dx, dy = np.broadcast_to(cell_size, 3)

        z_max = max(max(sur["SourceSurfaceElevation"].max(), sur["ReceiverGroupElevation"].max())
                    for sur in survey_list)
        if z_min >= z_max:
            raise ValueError
        nz = math.ceil((z_max - z_min) / dz)

        x_min = min(min(sur["SourceX"].min(), sur["GroupX"].min()) for sur in survey_list) - spatial_margin * dx
        x_max = max(max(sur["SourceX"].max(), sur["GroupX"].max()) for sur in survey_list) + spatial_margin * dx
        nx = math.ceil((x_max - x_min) / dx)

        y_min = min(min(sur["SourceY"].min(), sur["GroupY"].min()) for sur in survey_list) - spatial_margin * dy
        y_max = max(max(sur["SourceY"].max(), sur["GroupY"].max()) for sur in survey_list) + spatial_margin * dy
        ny = math.ceil((y_max - y_min) / dy)

        origin = (z_min, x_min, y_min)
        shape = (nz, nx, ny)
        cell_size = (dz, dx, dy)
        return cls(origin, shape, cell_size, survey=survey)
