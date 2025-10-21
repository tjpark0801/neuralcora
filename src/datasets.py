"""Refactored dataset utilities for NeuralCORA experiments.

This module now materializes a single C-contiguous float32 backing array per
dataset instance and removes all xarray work from ``__getitem__``. By default
the data are normalized using training statistics during construction; set
``normalize_on_gpu=True`` to keep raw values and apply normalization inside the
model (see the class docstring for a tiny example).

Recommended ``DataLoader`` settings::

    from torch.utils.data import DataLoader
    import os

    cpu_count = os.cpu_count() or 8
    num_workers = min(8, cpu_count)
    loader_opts = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,              # if using CUDA
        prefetch_factor=2 if num_workers > 0 else None,
    )

    train_loader = DataLoader(train_ds, **loader_opts)
    valid_loader = DataLoader(valid_ds, **{**loader_opts, "shuffle": False})
    test_loader  = DataLoader(test_ds,  **{**loader_opts, "shuffle": False})
"""

from __future__ import annotations

import atexit
import json
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from .data import ensure_lat_lon_dims

__all__ = ["VariableConfig", "CoraDataset"]

_CACHE_VERSION = 2


@dataclass(frozen=True)
class VariableConfig:
    """Configuration for a predictor variable.

    Parameters
    ----------
    name:
        Variable name in the NetCDF dataset.
    levels:
        Optional sequence of pressure levels to subset. ``None`` keeps all
        available levels or, for 2D fields, creates a singleton level.
    """

    name: str
    levels: Sequence[int] | None = None


def _get_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    for lat_name in ("lat", "latitude", "y"):
        if lat_name in ds.dims:
            break
    else:
        raise ValueError("Latitude dimension not found. Expected one of: lat, latitude, y.")

    for lon_name in ("lon", "longitude", "x"):
        if lon_name in ds.dims:
            break
    else:
        raise ValueError("Longitude dimension not found. Expected one of: lon, longitude, x.")

    return lat_name, lon_name


def _serialize_var_config(var_config: Mapping[str, Sequence[int] | None]) -> list[tuple[str, list[int] | None]]:
    serialized: list[tuple[str, list[int] | None]] = []
    for name, levels in OrderedDict(var_config).items():
        serialized.append((name, None if levels is None else list(levels)))
    return serialized


def _cache_meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _cache_is_compatible(
    meta: Mapping[str, Any],
    *,
    var_config: list[tuple[str, list[int] | None]],
    shape: tuple[int, int, int, int],
    lat_name: str,
    lon_name: str,
    float_dtype: str,
    level_values: Sequence[Any],
    mask_applied: bool,
    normalize_on_gpu: bool,
) -> bool:
    try:
        if meta.get("version") != _CACHE_VERSION:
            return False
        if tuple(meta.get("shape", ())) != shape:
            return False
        if meta.get("var_config") != var_config:
            return False
        if meta.get("lat_name") != lat_name or meta.get("lon_name") != lon_name:
            return False
        if meta.get("float_dtype") != float_dtype:
            return False
        if meta.get("level_values") != list(level_values):
            return False
        if bool(meta.get("mask_applied", False)) != mask_applied:
            return False
        if bool(meta.get("normalize_on_gpu", False)) != normalize_on_gpu:
            return False
    except Exception:
        return False
    return True


def _stack_vars_levels(
    ds: xr.Dataset,
    var_dict: Mapping[str, Sequence[int] | None],
    lat_name: str,
    lon_name: str,
) -> tuple[xr.DataArray, dict[str, slice], dict[str, np.ndarray | None]]:
    data_arrays: list[xr.DataArray] = []
    level_slices: dict[str, slice] = OrderedDict()
    level_coords: dict[str, np.ndarray | None] = {}
    offset = 0

    for var, levels in OrderedDict(var_dict).items():
        if var not in ds:
            raise KeyError(f"Variable {var!r} not found in dataset.")
        da = ds[var]
        if "level" in da.dims:
            da_sel = da if levels is None else da.sel(level=levels)
        else:
            if levels is not None:
                raise ValueError(f"Variable {var} does not have a 'level' dimension.")
            da_sel = da.expand_dims(level=[0])

        da_sel = da_sel.transpose("time", lat_name, lon_name, "level")
        data_arrays.append(da_sel)

        nlev = da_sel.sizes["level"]
        level_slices[var] = slice(offset, offset + nlev)
        level_coords[var] = None if levels is None else np.array(da_sel.level.values)
        offset += nlev

    data = xr.concat(data_arrays, dim="level")
    return data, level_slices, level_coords


def _compute_stats(
    data: xr.DataArray,
    lat_name: str,
    lon_name: str,
    *,
    mask: xr.DataArray | None = None,
    weights: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    if mask is not None:
        data = data.where(mask == 0)

    reduce_dims = ("time", lat_name, lon_name)

    if weights is not None:
        w = weights
        if w.dims == (lat_name,):
            w = w.broadcast_like(data.isel(time=0, level=0))
        if mask is not None:
            w = w.where(mask == 0, 0)
        mean = data.weighted(w).mean(reduce_dims, skipna=True)
        mean_sq = (data**2).weighted(w).mean(reduce_dims, skipna=True)
        var = (mean_sq - mean**2).clip(min=0.0)
        std = np.sqrt(var)
    else:
        mean = data.mean(reduce_dims, skipna=True)
        std = data.std(reduce_dims, skipna=True)

    mean = mean.fillna(0.0).astype("float32")
    std = std.fillna(1.0).astype("float32")
    std = xr.where(std == 0, 1.0, std)
    return mean, std


def _ensure_level_array(values: xr.DataArray | np.ndarray, level_values: np.ndarray, name: str) -> xr.DataArray:
    if isinstance(values, xr.DataArray):
        if "level" not in values.dims:
            raise ValueError(f"{name} must have a 'level' dimension.")
        arr = values.sel(level=level_values)
        data = np.asarray(arr.values, dtype=np.float32)
    else:
        data = np.asarray(values, dtype=np.float32)
        if data.shape != level_values.shape:
            raise ValueError(f"{name} has shape {data.shape}, expected shape {level_values.shape}.")
    return xr.DataArray(
        data,
        dims=("level",),
        coords={"level": level_values},
        name=name,
    )


class _LazyArrayView:
    """Lightweight view that exposes array-like metadata without materializing data."""

    __slots__ = ("_dataset", "_kind")

    def __init__(self, dataset: "CoraDataset", kind: str) -> None:
        self._dataset = dataset
        self._kind = kind

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def shape(self) -> tuple[int, ...]:
        ds = self._dataset
        lat, lon = ds.spatial_shape
        if self._kind == "input":
            if ds.flatten_inputs:
                return (len(ds), ds.input_channels, lat, lon)
            return (len(ds), ds.output_channels, ds.input_steps, lat, lon)
        return (len(ds), ds.output_channels, lat, lon)

    def __getitem__(self, index: int) -> np.ndarray:
        x, y = self._dataset[index]
        tensor = x if self._kind == "input" else y
        return tensor.detach().cpu().numpy()


class CoraDataset(Dataset):
    """PyTorch ``Dataset`` wrapping NeuralCORA grids with temporal context windows.

    The dataset materializes a single float32, C-contiguous backing array
    either in RAM or via an on-disk ``np.memmap``. By default the array holds
    normalized values computed from the training split. To keep raw values and
    normalize inside the model, initialise with ``normalize_on_gpu=True``::

        # model.__init__
        self.register_buffer("mean", torch.from_numpy(train_ds.mean_np.squeeze(0)))
        self.register_buffer("std",  torch.from_numpy(train_ds.std_np.squeeze(0)))

        # model.forward
        x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
    """

    def __init__(
        self,
        ds: xr.Dataset,
        var_config: Mapping[str, Sequence[int] | None],
        lead_time: int,
        *,
        input_steps: int = 1,
        flatten_inputs: bool = True,
        mean: xr.DataArray | None = None,
        std: xr.DataArray | None = None,
        load_into_memory: bool = True,
        mask: xr.DataArray | None = None,
        weights: xr.DataArray | None = None,
        float_dtype: str = "float32",
        cache_path: str | Path | None = None,
        normalize_on_gpu: bool = False,
    ) -> None:
        if "time" not in ds.dims:
            raise ValueError("Dataset must contain a 'time' dimension.")

        dtype_name = np.dtype(float_dtype).name
        if dtype_name != "float32":
            raise ValueError("CoraDataset fast path requires float32 dtype.")

        self.ds = ds
        self.var_config = OrderedDict(var_config)
        self.lead_time = int(lead_time)
        self.input_steps = max(1, int(input_steps))
        self.flatten_inputs = bool(flatten_inputs)
        self.float_dtype = "float32"
        self.normalize_on_gpu = bool(normalize_on_gpu)
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._temp_cache_path: Path | None = None
        self._temp_cleanup_registered = False

        ds = ensure_lat_lon_dims(ds)
        self.lat_name, self.lon_name = _get_lat_lon_names(ds)

        data, level_slices, level_coords = _stack_vars_levels(ds, self.var_config, self.lat_name, self.lon_name)
        if mask is not None:
            mask = ensure_lat_lon_dims(mask)
            data = data.where(mask == 0)

        self.level_slices = level_slices
        self.level_coords = level_coords

        level_values = np.asarray(data["level"].values)
        time_coord = np.asarray(data["time"].values)
        lat_coord = np.asarray(data[self.lat_name].values)
        lon_coord = np.asarray(data[self.lon_name].values)
        target_shape = (
            int(data.sizes["time"]),
            int(data.sizes["level"]),
            int(data.sizes[self.lat_name]),
            int(data.sizes[self.lon_name]),
        )
        serialized_var_config = _serialize_var_config(self.var_config)

        cache_meta: dict[str, Any] | None = None
        cached_array: np.memmap | None = None
        if self.cache_path is not None:
            meta_path = _cache_meta_path(self.cache_path)
            if self.cache_path.exists() and meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as fh:
                        cache_meta = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    cache_meta = None
                if cache_meta and not _cache_is_compatible(
                    cache_meta,
                    var_config=serialized_var_config,
                    shape=target_shape,
                    lat_name=self.lat_name,
                    lon_name=self.lon_name,
                    float_dtype=self.float_dtype,
                    level_values=level_values.tolist(),
                    mask_applied=mask is not None,
                    normalize_on_gpu=self.normalize_on_gpu,
                ):
                    cache_meta = None
                if cache_meta is not None:
                    try:
                        cached_array = np.lib.format.open_memmap(self.cache_path, mode="r")
                    except OSError:
                        cached_array = None

        if cached_array is not None and cache_meta is not None:
            mean_values = np.asarray(cache_meta["mean"], dtype=np.float32)
            std_values = np.asarray(cache_meta["std"], dtype=np.float32)
            self.mean = xr.DataArray(mean_values, dims=("level",), coords={"level": level_values}, name="mean")
            self.std = xr.DataArray(std_values, dims=("level",), coords={"level": level_values}, name="std")
            self._finalize_mean_std(level_values)

            if load_into_memory:
                array_data = np.array(cached_array, dtype=np.float32, copy=True, order="C")
            else:
                array_data = cached_array
            self._full_array: np.ndarray | np.memmap = array_data
        else:
            computed_mean: xr.DataArray | None = None
            computed_std: xr.DataArray | None = None
            if mean is None or std is None:
                computed_mean, computed_std = _compute_stats(
                    data,
                    self.lat_name,
                    self.lon_name,
                    mask=mask,
                    weights=weights,
                )
            resolved_mean = mean if mean is not None else computed_mean
            resolved_std = std if std is not None else computed_std
            if resolved_mean is None or resolved_std is None:
                raise RuntimeError("Failed to resolve mean/std statistics.")

            self.mean = resolved_mean
            self.std = resolved_std
            self._finalize_mean_std(level_values)

            if self.normalize_on_gpu:
                working = data.astype("float32")
            else:
                working = (data - self.mean) / self.std
                working = working.astype("float32")

            working = working.transpose("time", "level", self.lat_name, self.lon_name)

            total_bytes = int(np.prod(target_shape)) * np.dtype(np.float32).itemsize
            if load_into_memory and total_bytes > (2 * 1024**3):
                raise RuntimeError(
                    "load_into_memory=True would require materialising more than 2 GiB; "
                    "set load_into_memory=False to stream from disk."
                )

            dest_path = self.cache_path if self.cache_path is not None else self._create_temp_memmap_path()
            if self.cache_path is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            writer: np.memmap | None = None
            write_succeeded = False
            try:
                writer = np.lib.format.open_memmap(
                    dest_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=target_shape,
                )

                chunks = getattr(getattr(working, "data", None), "chunks", None)
                t_chunk: int | None = None
                if chunks:
                    try:
                        t_chunk = int(chunks[0][0])
                    except (TypeError, IndexError, ValueError):
                        t_chunk = None
                if not t_chunk or t_chunk <= 0:
                    t_chunk = 548

                for t0 in range(0, target_shape[0], t_chunk):
                    t1 = min(t0 + t_chunk, target_shape[0])
                    block_data = working.isel(time=slice(t0, t1)).data
                    if hasattr(block_data, "compute"):
                        block_data = block_data.compute()
                    block = np.asarray(block_data, dtype=np.float32, order="C")
                    block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
                    writer[t0:t1, :, :, :] = block
                    writer.flush()

                write_succeeded = True
            finally:
                if writer is not None:
                    writer.flush()
                    del writer
                if not write_succeeded:
                    try:
                        Path(dest_path).unlink()
                    except OSError:
                        pass

            reader = np.lib.format.open_memmap(dest_path, mode="r")
            if load_into_memory:
                try:
                    array_data = np.array(reader, dtype=np.float32, copy=True, order="C")
                finally:
                    del reader
                self._full_array = array_data
                if self.cache_path is None:
                    try:
                        Path(dest_path).unlink()
                    except OSError:
                        pass
            else:
                self._full_array = reader

            if self.cache_path is not None:
                self._write_cache_metadata(
                    array_shape=target_shape,
                    serialized_var_config=serialized_var_config,
                    level_values=level_values,
                    mask_applied=mask is not None,
                )

        if not isinstance(self._full_array, (np.ndarray, np.memmap)):
            raise TypeError("Backing array must be a NumPy array or np.memmap.")
        if isinstance(self._full_array, np.ndarray):
            self._full_array.setflags(write=False)

        self._validate_backing_array()

        self.output_channels = int(self._full_array.shape[1])
        self.channels = self.output_channels
        self._spatial_shape = (
            int(self._full_array.shape[2]),
            int(self._full_array.shape[3]),
        )

        total_steps = int(self._full_array.shape[0])
        min_required = (self.input_steps - 1) + self.lead_time + 1
        if total_steps < min_required:
            raise ValueError("Not enough timesteps for the requested input_steps and lead_time.")

        self.n_samples = total_steps - (self.input_steps - 1) - self.lead_time
        if self.n_samples <= 0:
            raise ValueError("Lead time and input_steps combination yields no samples.")

        if self.flatten_inputs:
            self.input_channels = self.output_channels * self.input_steps
        else:
            self.input_channels = self.output_channels

        target_start = self.input_steps - 1 + self.lead_time
        self.init_time = time_coord[self.input_steps - 1 : self.input_steps - 1 + self.n_samples]
        self.valid_time = time_coord[target_start : target_start + self.n_samples]
        self.lat = lat_coord
        self.lon = lon_coord

        # Rebuild mean/std metadata in case cache restored them before we created mean_np/std_np.
        self._finalize_mean_std(level_values)

        self.inputs = _LazyArrayView(self, "input")
        self.targets = _LazyArrayView(self, "target")

    def _create_temp_memmap_path(self) -> Path:
        fd, path = tempfile.mkstemp(prefix="cora_dataset_", suffix=".npy")
        os.close(fd)
        self._temp_cache_path = Path(path)
        if not self._temp_cleanup_registered:
            atexit.register(self._cleanup_temp_cache)
            self._temp_cleanup_registered = True
        return self._temp_cache_path

    def _cleanup_temp_cache(self) -> None:
        if self._temp_cache_path is None:
            return
        try:
            if self._temp_cache_path.exists():
                self._temp_cache_path.unlink()
        except OSError:
            pass

    def _write_cache_metadata(
        self,
        *,
        array_shape: tuple[int, int, int, int],
        serialized_var_config: list[tuple[str, list[int] | None]],
        level_values: np.ndarray,
        mask_applied: bool,
    ) -> None:
        if self.cache_path is None:
            return
        meta_path = _cache_meta_path(self.cache_path)
        meta = {
            "version": _CACHE_VERSION,
            "shape": array_shape,
            "var_config": serialized_var_config,
            "lat_name": self.lat_name,
            "lon_name": self.lon_name,
            "float_dtype": self.float_dtype,
            "level_values": level_values.tolist(),
            "mean": self.mean.values.astype(np.float32).tolist(),
            "std": self.std.values.astype(np.float32).tolist(),
            "mask_applied": bool(mask_applied),
            "normalize_on_gpu": self.normalize_on_gpu,
        }
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with meta_path.open("w", encoding="utf-8") as fh:
                json.dump(meta, fh)
        except OSError:
            pass

    def _validate_backing_array(self) -> None:
        arr = getattr(self, "_full_array", None)
        if arr is None:
            raise RuntimeError("Backing array was not initialised.")
        if arr.dtype != np.float32:
            raise TypeError(f"Backing array must be float32, got {arr.dtype}.")
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError("Backing array must be C-contiguous for zero-copy access.")

    def _finalize_mean_std(self, level_values: np.ndarray) -> None:
        mean_da = _ensure_level_array(self.mean, level_values, "mean")
        std_da = _ensure_level_array(self.std, level_values, "std")
        std_vals = np.asarray(std_da.values, dtype=np.float32)
        std_vals = np.where(std_vals == 0.0, 1.0, std_vals)
        std_da = xr.DataArray(std_vals, dims=("level",), coords={"level": level_values}, name="std")
        mean_vals = np.asarray(mean_da.values, dtype=np.float32)
        self.mean = mean_da
        self.std = std_da
        self.mean_np = mean_vals[None, :, None, None]
        self.std_np = std_vals[None, :, None, None]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.n_samples:
            raise IndexError(index)

        input_start = index
        input_end = index + self.input_steps
        target_index = index + self.input_steps - 1 + self.lead_time

        input_np = self._full_array[input_start:input_end]
        target_np = self._full_array[target_index]

        if self.flatten_inputs:
            input_np = input_np.reshape(self.input_channels, *self._spatial_shape)
        else:
            input_np = np.transpose(input_np, (1, 0, 2, 3))

        x = torch.from_numpy(input_np)
        y = torch.from_numpy(target_np)
        return x, y

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self._spatial_shape

    def unnormalize(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        return values * self.std_np + self.mean_np

    def to_xarray(self, values: np.ndarray, times: np.ndarray) -> xr.Dataset:
        values = np.asarray(values)
        data_arrays: list[xr.DataArray] = []
        for var, slc in self.level_slices.items():
            field = values[:, slc, :, :]
            if self.var_config[var] is None:
                data_arrays.append(
                    xr.DataArray(
                        field[:, 0],
                        dims=("time", self.lat_name, self.lon_name),
                        coords={
                            "time": times,
                            self.lat_name: self.lat,
                            self.lon_name: self.lon,
                        },
                        name=var,
                    )
                )
            else:
                level_coords = self.level_coords.get(var)
                if level_coords is None:
                    level_coords = np.arange(field.shape[1])
                data_arrays.append(
                    xr.DataArray(
                        np.moveaxis(field, 1, -1),
                        dims=("time", self.lat_name, self.lon_name, "level"),
                        coords={
                            "time": times,
                            self.lat_name: self.lat,
                            self.lon_name: self.lon,
                            "level": level_coords,
                        },
                        name=var,
                    )
                )
        return xr.merge(data_arrays)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    time = np.arange(6)
    lat = np.linspace(-1.0, 1.0, 2)
    lon = np.linspace(0.0, 1.5, 3)
    base = rng.standard_normal((6, 2, 3))
    ds = xr.Dataset(
        {"temp": (("time", "lat", "lon"), base)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    var_config = {"temp": None}

    dataset_flat = CoraDataset(
        ds,
        var_config,
        lead_time=1,
        input_steps=2,
        flatten_inputs=True,
        load_into_memory=True,
    )
    x_flat, y_flat = dataset_flat[0]
    assert x_flat.shape == (dataset_flat.input_channels, *dataset_flat.spatial_shape)
    assert y_flat.shape == (dataset_flat.output_channels, *dataset_flat.spatial_shape)
    assert isinstance(dataset_flat._full_array, (np.ndarray, np.memmap))

    dataset_seq = CoraDataset(
        ds,
        var_config,
        lead_time=1,
        input_steps=2,
        flatten_inputs=False,
        load_into_memory=True,
        normalize_on_gpu=True,
    )
    x_seq, y_seq = dataset_seq[0]
    assert x_seq.shape == (dataset_seq.output_channels, dataset_seq.input_steps, *dataset_seq.spatial_shape)
    assert y_seq.shape == (dataset_seq.output_channels, *dataset_seq.spatial_shape)
    assert isinstance(dataset_seq._full_array, (np.ndarray, np.memmap))

    print("Self-test passed.")
