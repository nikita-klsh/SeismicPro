import warnings
from textwrap import dedent

import cv2
import numpy as np
import pandas as pd
import polars as pl

from .validation import format_warning
from ..decorators import plotter


class Geometry:
    def __init__(self, bins, coords=None):
        bins = np.array(bins)
        if not np.issubdtype(bins.dtype, np.integer):
            raise TypeError
        if coords is not None:
            coords = np.array(coords)
            if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(bins):
                raise ValueError
        self.bins = bins
        self.coords = coords

    @property
    def has_coords(self):
        return self.coords is not None

    @property
    def n_bins(self):
        return len(self.bins)

    def __str__(self):
        raise NotImplementedError

    def info(self):
        print(self)

    def _validate(self, *args, **kwargs):
        _ = args, kwargs
        return None

    def validate(self, *args, **kwargs):
        msg = self._validate(*args, **kwargs)
        if msg is not None:
            warnings.warn(msg, RuntimeWarning)

    @plotter(figsize=(5, 5))
    def plot(self, ax=None, is_geographic=None, **kwargs):
        _ = ax, is_geographic, kwargs
        raise NotImplementedError


class Geometry2D(Geometry):
    def __init__(self, bins, coords=None, validate=True):
        super().__init__(bins, coords)
        if self.bins.ndim != 1:
            raise ValueError
        ix = np.argsort(self.bins)
        self.bins = self.bins[ix]

        # Calculate line mask
        self.line_mask = np.zeros(self.n_bins, dtype=np.uint8)
        self.line_mask[self.bins - self.line_origin] = 1

        # Calculate bin size if coords are available
        self.bin_size = None
        self.bin_sizes = None
        if self.has_coords:
            self.coords = self.coords[ix]
            self.bin_sizes = np.linalg.norm(np.diff(self.coords, axis=0), axis=-1) / np.diff(self.bins)
            self.bin_size = np.mean(self.bin_sizes)

        if validate:
            self.validate()

    @property
    def line_origin(self):
        return self.bins[0]

    @property
    def line_length(self):
        if not self.has_coords:
            return None
        return self.n_bins * self.bin_size

    def __str__(self):
        if not self.has_coords:
            msg = """Seismic line with undefined coordinates"""
        else:
            msg = f"""
            Line geometry:
            Bin size:                  {self.bin_size:.1f} m
            Line length:               {(self.line_length / 1000):.2f} km
            """
        return dedent(msg).strip()

    def _validate(self, cv_threshold=0.1):
        if not self.has_coords:
            return None
        cv = self.bin_sizes.std() / self.bin_size
        if cv < cv_threshold:
            return None
        return f"The line has highly variable bin sizes: its coefficient of variation is {cv:.2f}"

    def validate(self, cv_threshold=0.1):
        super().validate(cv_threshold=cv_threshold)

    @plotter(figsize=(5, 5))
    def plot(self, ax=None, is_geographic=None, color="gray", linewidth=3, alpha=0.9, **kwargs):
        if is_geographic is None:
            is_geographic = True
        if not is_geographic:
            raise ValueError("2D lines can be displayed only in geographic coordinate system")
        if not self.has_coords:
            raise ValueError
        ax.plot(self.coords[:, 0], self.coords[:, 1], color=color, linewidth=linewidth, alpha=alpha, **kwargs)


class Geometry3D(Geometry):
    def __init__(self, bins, coords=None, validate=True):
        super().__init__(bins, coords)
        if self.bins.ndim != 2 or self.bins.shape[1] != 2:
            raise ValueError

        # Calculate field mask and bin contours
        bins_min = self.bins.min(axis=0)
        bins_max = self.bins.max(axis=0)
        self.n_inline_bins, self.n_crossline_bins = bins_max - bins_min + 1
        self.field_origin = bins_min
        self.field_mask = np.zeros((self.n_inline_bins, self.n_crossline_bins), dtype=np.uint8)
        self.field_mask[self.bins[:, 0] - self.field_origin[0], self.bins[:, 1] - self.field_origin[1]] = 1
        self.bin_contours = cv2.findContours(self.field_mask.T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                             offset=self.field_origin)[0]

        # Infer field geometry if coords are available
        self.rotation_angle = None
        self.bin_size = None
        self.is_reflected = None
        self.bias = None
        self._bins_to_coords = None
        self._coords_to_bins = None
        self.fit_loss = None
        self.geographic_contours = None
        if self.has_coords:
            transform = self.fit_scaled_rigid_transform(self.bins, self.coords)
            self.rotation_angle = transform[0]
            self.bin_size = np.abs(transform[1])
            self.is_reflected = transform[1] < 0
            self.bias = transform[2]
            self._bins_to_coords = transform[3]
            self._coords_to_bins = transform[4]
            self.fit_loss = transform[5]
            self.geographic_contours = tuple(self.bins_to_coords(contour[:, 0])[:, None].astype(np.float32)
                                             for contour in self.bin_contours)

        if validate:
            self.validate()

    @property
    def inline_length(self):
        if not self.has_coords:
            return None
        return self.n_inline_bins * self.bin_size[0]

    @property
    def crossline_length(self):
        if not self.has_coords:
            return None
        return self.n_crossline_bins * self.bin_size[1]

    @property
    def area(self):
        if not self.has_coords:
            return None
        return self.n_bins * self.bin_size[0] * self.bin_size[1]

    @property
    def perimeter(self):
        if not self.has_coords:
            return None
        return sum(cv2.arcLength(contour, closed=True) for contour in self.geographic_contours)

    @staticmethod
    def fit_scaled_rigid_transform(src, dst):
        """See https://math.stackexchange.com/questions/3955634/ for detailed algorithm derivation."""
        src = np.require(src, dtype=np.float64)
        dst = np.require(dst, dtype=np.float64)

        src_centered = src - src.mean(axis=0)
        dst_centered = dst - dst.mean(axis=0)

        # Use mean instead of sum for numerical stability
        c1 = np.square(src_centered[:, 0]).mean()
        c2 = np.square(src_centered[:, 1]).mean()
        b11 = -2 * (src_centered[:, 0] * dst_centered[:, 0]).mean()
        b12 = -2 * (src_centered[:, 0] * dst_centered[:, 1]).mean()
        b21 = -2 * (src_centered[:, 1] * dst_centered[:, 0]).mean()
        b22 = -2 * (src_centered[:, 1] * dst_centered[:, 1]).mean()

        k_num = b11**2 * c2 + b22**2 * c1 - b12**2 * c2 - b21**2 * c1
        k_denom = b12 * b11 * c2 - b21 * b22 * c1
        if np.isclose(k_denom, 0):
            # Degenerate case: points of src or dst lie on a line in 2D
            src_diff = np.diff(src_centered, axis=0)
            src_diff_normed = src_diff / np.linalg.norm(src_diff, axis=1, keepdims=True)
            dst_diff = np.diff(dst_centered, axis=0)
            dst_diff_normed = dst_diff / np.linalg.norm(dst_diff, axis=1, keepdims=True)
            dot = np.clip((src_diff_normed * dst_diff_normed).sum(axis=1), -1, 1)
            angle = np.rad2deg(np.arccos(dot).mean())
            if angle >= 90:
                angle -= 180
            angles = [-angle, angle]
        else:
            k = k_num / k_denom
            r1 = np.sqrt(2 / (k**2 + 4 - k * np.sqrt(k**2 + 4)))
            r2 = np.sqrt(2 / (k**2 + 4 + k * np.sqrt(k**2 + 4)))
            angles = [np.arctan2(r1, r2), np.arctan2(r1, -r2), np.arctan2(-r1, r2), np.arctan2(-r1, -r2)]
            if not np.isclose(r1, r2):
                angles += [np.arctan2(r2, r1), np.arctan2(r2, -r1), np.arctan2(-r2, r1), np.arctan2(-r2, -r1)]
            angles = [np.rad2deg(angle) for angle in angles]
            angles = [angle for angle in angles if -90 <= angle < 90]

        transforms_params = []
        for angle in angles:
            r1 = np.cos(np.deg2rad(angle))
            r2 = np.sin(np.deg2rad(angle))
            rotation = np.array([[r1, -r2], [r2, r1]])

            scale_x = 0 if np.isclose(c1, 0) else (-b11 * r1 - b12 * r2) / (2 * c1)
            scale_y = 0 if np.isclose(c2, 0) else (b21 * r2 - b22 * r1) / (2 * c2)
            scale = np.array([scale_x, scale_y])

            err = dst - (src * scale) @ rotation.T
            bias = err.mean(axis=0)
            loss = np.mean(np.sqrt(np.square(err - bias).sum(axis=1)))
            transforms_params.append((angle, rotation, scale, bias, loss))

        angle, rotation, scale, bias, loss = min(transforms_params, key=lambda x: x[-1])
        src_to_dst = lambda src: (src * scale) @ rotation.T + bias
        dst_to_src = lambda dst: ((dst - bias) @ rotation) / scale
        return angle, scale, bias, src_to_dst, dst_to_src, loss

    @staticmethod
    def _cast_coords(coords, transformer):
        if transformer is None:
            raise ValueError("Geometry does not store coordinates data")
        coords = np.array(coords)
        is_coords_1d = coords.ndim == 1
        coords = np.atleast_2d(coords)
        transformed_coords = transformer(coords)
        if is_coords_1d:
            return transformed_coords[0]
        return transformed_coords

    def coords_to_bins(self, coords):
        return self._cast_coords(coords, self._coords_to_bins)

    def bins_to_coords(self, bins):
        return self._cast_coords(bins, self._bins_to_coords)

    @staticmethod
    def _dist_to_contours(coords, contours):
        coords = np.array(coords, dtype=np.float32)
        is_coords_1d = coords.ndim == 1
        coords = np.atleast_2d(coords)
        dist = np.empty(len(coords), dtype=np.float32)
        for i, coord in enumerate(coords):
            dists = [cv2.pointPolygonTest(contour, coord, measureDist=True) for contour in contours]
            dist[i] = dists[np.abs(dists).argmin()]
        if is_coords_1d:
            return dist[0]
        return dist

    def dist_to_geographic_contours(self, coords):
        if not self.has_coords:
            raise ValueError("Geometry does not store coordinates data")
        return self._dist_to_contours(coords, self.geographic_contours)

    def dist_to_bin_contours(self, bins):
        return self._dist_to_contours(bins, self.bin_contours)

    def __str__(self):
        if not self.has_coords:
            msg = """3D seismic geometry with undefined coordinates"""
        else:
            msg = f"""
            Field geometry:
            Inline bin size:           {self.bin_size[0]:.1f} m
            Crossline bin size:        {self.bin_size[1]:.1f} m
            Inline length:             {(self.inline_length / 1000):.2f} km
            Crossline length:          {(self.crossline_length / 1000):.2f} km
            Perimeter:                 {(self.perimeter / 1000):.2f} km
            Area:                      {(self.area / 1000**2):.2f} km^2
            """
        return dedent(msg).strip()

    def _validate(self, threshold=None):
        if not self.has_coords:
            return None
        if threshold is None:
            threshold = (self.bin_size[0]**2 + self.bin_size[1]**2)**0.5 / 2
        if self.fit_loss < threshold:
            return None
        return "The transform from bins to coords is poorly fit"

    def validate(self, threshold=None):
        super().validate(threshold=threshold)

    @plotter(figsize=(5, 5))
    def plot(self, ax=None, is_geographic=None, color="gray", edgecolor="black", alpha=0.5, **kwargs):
        if is_geographic is None:
            is_geographic = self.has_coords
        if is_geographic and not self.has_coords:
            raise ValueError("Geometry can be displayed in geographic coordinate system only if "
                             "coordinates were passed")
        contours = self.geographic_contours if is_geographic else self.bin_contours
        for contour in contours:
            ax.fill(contour[:, 0, 0], contour[:, 0, 1], color=color, edgecolor=edgecolor, alpha=alpha, **kwargs)


def infer_geometry(bin_headers, validate=True, warn_width=80):
    if isinstance(bin_headers, pd.DataFrame):
        bin_headers = pl.from_pandas(bin_headers)
    if not isinstance(bin_headers, pl.DataFrame):
        raise TypeError
    bin_headers = bin_headers.to_pandas(use_pyarrow_extension_array=False)  # cast to numpy types

    if "CDP" in bin_headers:
        bins = bin_headers["CDP"].to_numpy()
        geometry_type = Geometry2D
    elif "INLINE_3D" in bin_headers and "CROSSLINE_3D" in bin_headers:
        bins = bin_headers[["INLINE_3D", "CROSSLINE_3D"]].to_numpy()
        geometry_type = Geometry3D
    else:
        raise ValueError

    msg_list = []

    bins_int = np.require(bins, np.int32)
    if validate and not np.allclose(bins, bins_int):
        msg_list.append("Bin index can not be safely cast to integer type")

    coords = bin_headers.get(["CDP_X", "CDP_Y"])
    if coords is not None:
        coords = coords.to_numpy()

    geometry = geometry_type(bins_int, coords, validate=False)
    if validate:
        msg = geometry._validate()
        if msg is not None:
            msg_list.append(msg)

    if validate and msg_list:
        warn_str = "Survey geometry has the following problems:"
        warn_str = format_warning(warn_str, msg_list, width=warn_width)
        return geometry, warn_str
    return geometry, None
