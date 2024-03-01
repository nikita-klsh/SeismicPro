from string import Formatter
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from batchflow import Batch, NamedExpression
from batchflow.decorators import action, inbatch_parallel

from ..gather import Gather, CroppedGather
from ..gather.utils.crop_utils import make_origins
from ..velocity_spectrum import VerticalVelocitySpectrum, ResidualVelocitySpectrum, SlantStack
from ..field import Field
from ..decorators import create_batch_methods
from ..utils import to_list, align_src_dst, as_dict, save_figure


@create_batch_methods(Gather, CroppedGather, VerticalVelocitySpectrum, ResidualVelocitySpectrum, SlantStack)
class SeismicBatch(Batch):
    def __init__(self, pos):
        super().__init__(pos)

    def init_component(self, *args, dst=None, **kwargs):
        """Create and preallocate new attributes with names listed in `dst` if they don't exist and return ordinal
        numbers of batch items. This method is typically used as a default `init` function in `inbatch_parallel`
        decorator."""
        _ = args, kwargs
        dst = [] if dst is None else to_list(dst)
        for comp in dst:
            if self.components is None or comp not in self.components:
                self.add_components(comp, init=self.array_of_nones)
        return np.arange(len(self))

    def init_coef_component(self, *args, dst=None, dst_coefs=None, **kwargs):
        """Create and preallocate new attributes with names listed in `dst` and `dst_coefs` if they don't exist and
        return ordinal numbers of batch items. This method is used as a default `init` for `apply_agc` method."""
        dst_coefs = [] if dst_coefs is None else to_list(dst_coefs)
        dst = [] if dst is None else to_list(dst)
        return self.init_component(*args, dst=dst+dst_coefs, **kwargs)

    @property
    def flat_indices(self):
        """np.ndarray: Unique identifiers of seismic gathers in the batch flattened into a 1d array."""
        # TODO: handle for compound survey case
        return self.indices

    @action
    def update_field(self, field, src):
        if not isinstance(field, Field):
            raise ValueError("Only a Field instance can be updated")
        field.update(getattr(self, src))
        return self

    @action
    def evaluate_field(self, field, src, dst):
        _ = self.init_component(dst=dst)
        field_items = field([item.coords for item in getattr(self, src)])
        for i, item in enumerate(field_items):
            setattr(self[i], dst, item)
        return self

    @action
    def make_model_inputs(self, src, dst, mode='c', axis=0, expand_dims_axis=None):
        data = getattr(self, src) if isinstance(src, str) else src
        func = {'c': np.concatenate, 's': np.stack}.get(mode)
        if func is None:
            raise ValueError(f"Unknown mode '{mode}', must be either 'c' or 's'")
        data = func(data, axis=axis)

        if expand_dims_axis is not None:
            data = np.expand_dims(data, axis=expand_dims_axis)
        setattr(self, dst, data)
        return self

    @action(no_eval='dst')
    def split_model_outputs(self, src, dst, shapes):
        data = getattr(self, src) if isinstance(src, str) else src
        shapes = np.cumsum(shapes)
        if shapes[-1] != len(data):
            raise ValueError("Data length must match the sum of shapes passed")
        split_data = np.split(data, shapes[:-1])

        if isinstance(dst, str):
            setattr(self, dst, split_data)
        elif isinstance(dst, NamedExpression):
            dst.set(value=split_data)
        else:
            raise ValueError(f"dst must be either `str` or `NamedExpression`, not {type(dst)}.")
        return self

    @action
    @inbatch_parallel(init="init_component", target="threads")
    def crop(self, pos, src, origins, crop_shape, dst=None, joint=True, n_crops=1, stride=None, **kwargs):
        src_list, dst_list = align_src_dst(src, dst)

        if joint:
            src_shapes = set()
            src_types = set()

            for src in src_list:  # pylint: disable=redefined-argument-from-local
                src_obj = getattr(self, src)[pos]
                src_types.add(type(src_obj))
                src_shapes.add(src_obj.shape)

            if len(src_types) > 1:
                raise TypeError("If joint is True, all src components must be of the same type.")
            if len(src_shapes) > 1:
                raise ValueError("If joint is True, all src components must have the same shape.")
            data_shape = src_shapes.pop()
            origins = make_origins(origins, data_shape, crop_shape, n_crops, stride)

        for src, dst in zip(src_list, dst_list):  # pylint: disable=redefined-argument-from-local
            src_obj = getattr(self, src)[pos]
            src_cropped = src_obj.crop(origins, crop_shape, n_crops, stride, **kwargs)
            setattr(self[pos], dst, src_cropped)

        return self

    @action
    @inbatch_parallel(init="init_coef_component", target="threads")
    def apply_agc(self, pos, src, dst=None, dst_coefs=None, window_size=250, mode='rms'):
        src_list, dst_list = align_src_dst(src, dst)
        dst_coefs_list = to_list(dst_coefs) if dst_coefs is not None else [None] * len(dst_list)
        if len(dst_coefs_list) != len(dst_list):
            raise ValueError("dst_coefs and dst should have the same length.")

        # pylint: disable-next=redefined-argument-from-local
        for src, dst_coef, dst in zip(src_list, dst_coefs_list, dst_list):
            src_obj = getattr(self, src)[pos]
            src_obj = src_obj.copy() if src != dst else src_obj
            return_coefs = dst_coef is not None
            results = src_obj.apply_agc(window_size=window_size, mode=mode, return_coefs=return_coefs)
            if return_coefs:
                setattr(self[pos], dst, results[0])
                setattr(self[pos], dst_coef, results[1])
            else:
                setattr(self[pos], dst, results)
        return self

    @action
    @inbatch_parallel(init="init_component", target="threads")
    def undo_agc(self, pos, src, src_coefs, dst=None):
        src_list, dst_list = align_src_dst(src, dst)
        src_coefs_list = to_list(src_coefs)

        if len(src_list) != len(src_coefs_list):
            raise ValueError("The length of `src_coefs` must match the length of `src`")

        # pylint: disable-next=redefined-argument-from-local
        for src, coef, dst in zip(src_list, src_coefs_list, dst_list):
            src_obj = getattr(self, src)[pos]
            src_coef = getattr(self, coef)[pos]

            src_obj = src_obj.copy() if src != dst else src_obj
            src_noagc = src_obj.undo_agc(coefs_gather=src_coef)
            setattr(self[pos], dst, src_noagc)
        return self

    @staticmethod
    def _unpack_args(args, batch_item):
        """Replace all names of batch components in `args` with corresponding values from `batch_item`. """
        if not isinstance(args, (list, tuple, str)):
            return args

        unpacked_args = [getattr(batch_item, val) if isinstance(val, str) and val in batch_item.components else val
                         for val in to_list(args)]
        if isinstance(args, str):
            return unpacked_args[0]
        return unpacked_args

    @action  # pylint: disable-next=too-many-statements
    def plot(self, src, src_kwargs=None, max_width=20, title="{src}: {index}", save_to=None, **common_kwargs):
        # Construct a list of plot kwargs for each component in src
        src_list = to_list(src)
        if src_kwargs is None:
            src_kwargs = [{} for _ in range(len(src_list))]
        elif isinstance(src_kwargs, dict):
            src_kwargs = {src: src_kwargs[keys] for keys in src_kwargs for src in to_list(keys)}
            src_kwargs = [src_kwargs.get(src, {}) for src in src_list]
        else:
            src_kwargs = to_list(src_kwargs)
            if len(src_list) != len(src_kwargs):
                raise ValueError("The length of src_kwargs must match the length of src")

        # Construct a grid of plotters with shape (len(self), len(src_list)) for each of the subplots
        plotters = [[] for _ in range(len(self))]
        for src, kwargs in zip(src_list, src_kwargs):  # pylint: disable=redefined-argument-from-local
            # Merge src kwargs with common kwargs and defaults
            plotter_params = getattr(getattr(self, src)[0].plot, "method_params", {}).get("plotter")
            if plotter_params is None:
                raise ValueError("plot method of each component in src must be decorated with plotter")
            kwargs = {"figsize": plotter_params["figsize"], "title": title, **common_kwargs, **kwargs}

            # Scale subplot figsize if its width is greater than max_width
            width, height = kwargs.pop("figsize")
            if width > max_width:
                height = height * max_width / width
                width = max_width

            title_template = kwargs.pop("title")
            args_to_unpack = set(to_list(plotter_params["args_to_unpack"]))

            for i, index in enumerate(self.flat_indices):
                # Unpack required plotter arguments by getting the value of specified component with given index
                unpacked_args = {}
                for arg_name in args_to_unpack & kwargs.keys():
                    arg_val = kwargs[arg_name]
                    if isinstance(arg_val, dict) and arg_name in arg_val:
                        arg_val[arg_name] = self._unpack_args(arg_val[arg_name], self[i])
                    else:
                        arg_val = self._unpack_args(arg_val, self[i])
                    unpacked_args[arg_name] = arg_val

                # Format subplot title
                if title_template is not None:
                    src_title = as_dict(title_template, key='label')
                    label = src_title.pop("label")
                    format_names = {name for _, name, _, _ in Formatter().parse(label) if name is not None}
                    format_kwargs = {name: src_title.pop(name) for name in format_names if name in src_title}
                    src_title["label"] = label.format(src=src, index=index, **format_kwargs)
                    kwargs["title"] = src_title

                # Create subplotter config
                subplot_config = {
                    "plotter": partial(getattr(self, src)[i].plot, **{**kwargs, **unpacked_args}),
                    "height": height,
                    "width": width,
                }
                plotters[i].append(subplot_config)

        # Flatten all the subplots into a row if a single component was specified
        if len(src_list) == 1:
            plotters = [sum(plotters, [])]

        # Wrap lines of subplots wider than max_width
        split_pos = []
        curr_width = 0
        for i, plotter in enumerate(plotters[0]):
            curr_width += plotter["width"]
            if curr_width > max_width:
                split_pos.append(i)
                curr_width = plotter["width"]
        plotters = sum([np.split(plotters_row, split_pos) for plotters_row in plotters], [])

        # Define axes layout and perform plotting
        fig_width = max(sum(plotter["width"] for plotter in plotters_row) for plotters_row in plotters)
        row_heights = [max(plotter["height"] for plotter in plotters_row) for plotters_row in plotters]
        fig = plt.figure(figsize=(fig_width, sum(row_heights)), tight_layout=True)
        gridspecs = fig.add_gridspec(len(plotters), 1, height_ratios=row_heights)

        for gridspecs_row, plotters_row in zip(gridspecs, plotters):
            n_cols = len(plotters_row)
            col_widths = [plotter["width"] for plotter in plotters_row]

            # Create a dummy axis if row width is less than fig_width in order to avoid row stretching
            if fig_width > sum(col_widths):
                col_widths.append(fig_width - sum(col_widths))
                n_cols += 1

            # Create a gridspec for the current row
            gridspecs_col = gridspecs_row.subgridspec(1, n_cols, width_ratios=col_widths)
            for gridspec, plotter in zip(gridspecs_col, plotters_row):
                plotter["plotter"](ax=fig.add_subplot(gridspec))

        if save_to is not None:
            save_kwargs = as_dict(save_to, key="fname")
            save_figure(fig, **save_kwargs)
        return self
