"""Evaluation metrics for NeuralCORA forecasts."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .data import ensure_lat_lon_dims

__all__ = ["rename_for_score", "compute_weighted_rmse"]


def rename_for_score(da: xr.DataArray) -> xr.DataArray:
    """Return a copy with CF-compliant latitude/longitude names for scoring."""
    da = ensure_lat_lon_dims(da)
    rename: dict[str, str] = {}
    if "lat" in da.dims and "latitude" not in da.dims:
        rename["lat"] = "latitude"
    if "lon" in da.dims and "longitude" not in da.dims:
        rename["lon"] = "longitude"
    return da.rename(rename) if rename else da


def compute_weighted_rmse(
    forecast: xr.DataArray | xr.Dataset,
    truth: xr.DataArray,
    *,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute RMSE for matching forecast and truth fields."""

    if isinstance(forecast, xr.Dataset):
        forecast = next(iter(forecast.data_vars.values()))

    forecast = ensure_lat_lon_dims(forecast)
    truth = ensure_lat_lon_dims(truth)

    if "valid_time" in forecast.coords:
        obs = truth.sel(time=forecast["valid_time"])
    elif "time" in forecast.dims:
        obs = truth
    else:
        forecast = forecast.expand_dims(time=truth.time)
        obs = truth

    sq_err = (forecast - obs) ** 2

    if weights is not None:
        weights = ensure_lat_lon_dims(weights)
        norm_weights = weights / weights.sum()
        mse = sq_err.weighted(norm_weights).mean(dim=("latitude", "longitude"))
    else:
        spatial_dims = [dim for dim in sq_err.dims if dim not in {"time", "lead_time"}]
        mse = sq_err.mean(dim=spatial_dims)

    rmse_per_fc = np.sqrt(mse)

    if "time" in rmse_per_fc.dims:
        rmse = rmse_per_fc.mean(dim="time")
    else:
        rmse = rmse_per_fc

    rmse.name = "RMSE"
    return rmse
