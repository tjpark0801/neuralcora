"""Dataset utilities for NeuralCORA experiments."""

from __future__ import annotations

import json
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
) -> bool:
    try:
        if meta.get("version") != 1:
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
    """PyTorch ``Dataset`` wrapping NeuralCORA grids with temporal context windows."""

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
    ) -> None:
        if "time" not in ds.dims:
            raise ValueError("Dataset must contain a 'time' dimension.")

        self.ds = ds
        self.var_config = OrderedDict(var_config)
        self.lead_time = int(lead_time)
        self.input_steps = max(1, int(input_steps))
        self.flatten_inputs = bool(flatten_inputs)
        self.float_dtype = float_dtype
        self.cache_path = Path(cache_path) if cache_path is not None else None

        ds = ensure_lat_lon_dims(ds)
        self.lat_name, self.lon_name = _get_lat_lon_names(ds)

        data, level_slices, level_coords = _stack_vars_levels(ds, self.var_config, self.lat_name, self.lon_name)
        self.level_slices = level_slices
        self.level_coords = level_coords
        level_values = np.asarray(data["level"].values)
        target_shape = (
            int(data.sizes["time"]),
            int(data.sizes["level"]),
            int(data.sizes[self.lat_name]),
            int(data.sizes[self.lon_name]),
        )
        serialized_var_config = _serialize_var_config(self.var_config)

        cache_meta: dict[str, Any] | None = None
        cached_array: np.ndarray | None = None
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
                ):
                    cache_meta = None
                if cache_meta is not None:
                    try:
                        cached_array = np.lib.format.open_memmap(
                            self.cache_path, mode="r"
                        )
                    except OSError:
                        cached_array = None

        if mask is not None:
            mask = ensure_lat_lon_dims(mask)
            data = data.where(mask == 0)

        cache_used = cached_array is not None and cache_meta is not None

        if cache_used:
            time_coord = np.asarray(data["time"].values)
            lat_coord = np.asarray(data[self.lat_name].values)
            lon_coord = np.asarray(data[self.lon_name].values)

            base_array = cached_array
            if base_array.dtype != np.dtype(self.float_dtype):
                base_array = base_array.astype(self.float_dtype)

            if load_into_memory:
                array_data = np.array(base_array, dtype=self.float_dtype, copy=True)
                array_for_fast_path: np.ndarray | None = array_data
            else:
                array_data = base_array
                array_for_fast_path = base_array if isinstance(base_array, np.memmap) else None

            normalized = xr.DataArray(
                array_data,
                dims=("time", "level", self.lat_name, self.lon_name),
                coords={
                    "time": time_coord,
                    "level": level_values,
                    self.lat_name: lat_coord,
                    self.lon_name: lon_coord,
                },
                name="normalized",
            )

            mean_values = np.asarray(cache_meta["mean"], dtype=self.float_dtype)
            std_values = np.asarray(cache_meta["std"], dtype=self.float_dtype)
            self.mean = xr.DataArray(mean_values, dims=("level",), coords={"level": level_values})
            self.std = xr.DataArray(std_values, dims=("level",), coords={"level": level_values})
            self._full_array = (
                np.nan_to_num(array_for_fast_path, nan=0.0, posinf=0.0, neginf=0.0)
                if array_for_fast_path is not None and not isinstance(array_for_fast_path, np.memmap)
                else array_for_fast_path
            )
        else:
            if mean is None or std is None:
                computed_mean, computed_std = _compute_stats(
                    data,
                    self.lat_name,
                    self.lon_name,
                    mask=mask,
                    weights=weights,
                )
                self.mean = computed_mean if mean is None else mean
                self.std = computed_std if std is None else std
            else:
                self.mean, self.std = mean, std

            if hasattr(self.mean, "compute"):
                self.mean = self.mean.compute()
            if hasattr(self.std, "compute"):
                self.std = self.std.compute()

            normalized = (data - self.mean) / self.std
            normalized = normalized.astype(self.float_dtype)
            normalized = normalized.transpose("time", "level", self.lat_name, self.lon_name)

            if load_into_memory:
                normalized = normalized.load()
                array_data = np.asarray(normalized.data)
                array_data = array_data.astype(self.float_dtype, copy=False)
                array_data = np.nan_to_num(array_data, nan=0.0, posinf=0.0, neginf=0.0)
                self._full_array = array_data
            else:
                self._full_array = None

        self._data = normalized
        if load_into_memory and self._full_array is not None:
            normalized = xr.DataArray(
                self._full_array,
                dims=("time", "level", self.lat_name, self.lon_name),
                coords={
                    "time": normalized["time"].values,
                    "level": normalized["level"].values,
                    self.lat_name: normalized[self.lat_name].values,
                    self.lon_name: normalized[self.lon_name].values,
                },
                name="normalized",
            )
            self._data = normalized

        if (
            self.cache_path is not None
            and not cache_used
            and self._full_array is not None
        ):
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                memmap = np.lib.format.open_memmap(
                    self.cache_path,
                    mode="w+",
                    dtype=self._full_array.dtype,
                    shape=self._full_array.shape,
                )
                memmap[...] = self._full_array
                memmap.flush()
                del memmap
                meta = {
                    "version": 1,
                    "shape": target_shape,
                    "var_config": serialized_var_config,
                    "lat_name": self.lat_name,
                    "lon_name": self.lon_name,
                    "float_dtype": self.float_dtype,
                    "level_values": level_values.tolist(),
                    "mean": self.mean.values.astype(self.float_dtype).tolist(),
                    "std": self.std.values.astype(self.float_dtype).tolist(),
                    "mask_applied": mask is not None,
                }
                meta_path = _cache_meta_path(self.cache_path)
                with meta_path.open("w", encoding="utf-8") as fh:
                    json.dump(meta, fh)
            except OSError:
                # Silently skip cache write failures to keep training running.
                pass

        self.output_channels = int(self._data.sizes["level"])
        self.channels = self.output_channels
        self._spatial_shape = (
            int(self._data.sizes[self.lat_name]),
            int(self._data.sizes[self.lon_name]),
        )

        total_steps = int(normalized.sizes["time"])
        min_required = (self.input_steps - 1) + self.lead_time + 1
        if total_steps < min_required:
            raise ValueError(
                "Not enough timesteps for the requested input_steps and lead_time."
            )

        self.n_samples = total_steps - (self.input_steps - 1) - self.lead_time
        if self.n_samples <= 0:
            raise ValueError("Lead time and input_steps combination yields no samples.")

        if self.flatten_inputs:
            self.input_channels = self.output_channels * self.input_steps
        else:
            self.input_channels = self.output_channels

        target_start = self.input_steps - 1 + self.lead_time
        time_coords = normalized["time"].values
        self.init_time = time_coords[self.input_steps - 1 : self.input_steps - 1 + self.n_samples]
        self.valid_time = time_coords[target_start : target_start + self.n_samples]
        self.lat = normalized[self.lat_name].values
        self.lon = normalized[self.lon_name].values

        mean_vals = np.nan_to_num(self.mean.values.astype(self.float_dtype), nan=0.0)
        std_vals = np.nan_to_num(self.std.values.astype(self.float_dtype), nan=1.0)
        std_vals = np.where(std_vals == 0.0, 1.0, std_vals)
        self.mean_np = mean_vals[None, :, None, None]
        self.std_np = std_vals[None, :, None, None]

        self.inputs = _LazyArrayView(self, "input")
        self.targets = _LazyArrayView(self, "target")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.n_samples:
            raise IndexError(index)

        input_start = index
        input_end = index + self.input_steps
        target_index = index + self.input_steps - 1 + self.lead_time

        if self._full_array is not None:
            input_np = self._full_array[input_start:input_end]
            target_np = self._full_array[target_index]
        else:
            input_slice = self._data.isel(time=slice(input_start, input_end))
            target_slice = self._data.isel(time=target_index)
            input_np = input_slice.to_numpy()
            target_np = target_slice.to_numpy()

            input_np = input_np.astype(self.float_dtype, copy=False)
            target_np = target_np.astype(self.float_dtype, copy=False)

            input_np = np.nan_to_num(input_np, nan=0.0, posinf=0.0, neginf=0.0)
            target_np = np.nan_to_num(target_np, nan=0.0, posinf=0.0, neginf=0.0)

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
