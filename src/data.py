"""Convenient helpers for working with NeuralCORA NetCDF datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr

__all__ = [
    "parse_period",
    "open_period",
    "collect_year_files",
    "ensure_lat_lon_dims",
    "subset_by_years",
    "load_land_mask",
]


def _safe_end_of_day(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the inclusive end-of-day timestamp when no time component is provided."""
    if ts.time() == pd.Timestamp(0).time():
        return ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def parse_period(spec: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse a period specification into inclusive ``(start, end)`` timestamps.

    Accepted formats:
    - ``YYYY-YYYY``
    - ``YYYYMMDD-YYYYMMDD``
    - ``YYYYMMDDHH-YYYYMMDDHH``
    """
    spec = spec.strip()
    if "-" not in spec:
        raise ValueError("Period must be a range like '1998-2020' or '19980101-20101231'.")

    left, right = spec.split("-", 1)

    if len(left) == 4 and len(right) == 4:
        start = pd.to_datetime(f"{left}-01-01 00:00:00")
        end = pd.to_datetime(f"{right}-12-31 23:59:59")
        return start, end

    def _parse_side(value: str) -> pd.Timestamp:
        if len(value) == 8:
            return pd.to_datetime(value, format="%Y%m%d")
        if len(value) == 10:
            return pd.to_datetime(value, format="%Y%m%d%H")
        return pd.to_datetime(value)

    start = _parse_side(left)
    end = _parse_side(right)
    end = _safe_end_of_day(end)
    return start, end


def _year_template_path(data_dir: Path, year: int, template: str) -> Path:
    return data_dir / template.format(year=year)


def collect_year_files(
    data_dir: str | Path,
    start_year: int,
    end_year: int,
    *,
    template: str = "NY_{year}_180_360.nc",
) -> list[Path]:
    """Collect existing yearly files within the given inclusive range."""
    data_dir = Path(data_dir).expanduser().resolve()
    files: list[Path] = []
    for year in range(start_year, end_year + 1):
        candidate = _year_template_path(data_dir, year, template)
        if candidate.exists():
            files.append(candidate)
    if not files:
        raise FileNotFoundError(
            f"No yearly files found between {start_year} and {end_year} in {data_dir}."
        )
    return files


def open_period(
    period_spec: str,
    data_dir: str | Path,
    *,
    template: str = "NY_{year}_180_360.nc",
    chunks: dict | None = None,
    parallel: bool = False,
    engine: str | None = "netcdf4",
) -> xr.Dataset:
    """Open and concatenate all NetCDF files covering the given period specification."""
    start_ts, end_ts = parse_period(period_spec)
    files = collect_year_files(data_dir, start_ts.year, end_ts.year, template=template)
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        combine="by_coords",
        parallel=parallel,
        chunks=chunks,
        engine=engine,
    )
    ds = ds.sel(time=slice(start_ts, end_ts))
    if ds.sizes.get("time", 0) == 0:
        raise ValueError(f"No data within requested window {start_ts} to {end_ts}.")
    return ds


def ensure_lat_lon_dims(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Rename latitude/longitude dimensions to ``lat``/``lon`` if needed."""
    rename: dict[str, str] = {}
    if "latitude" in ds.dims and "lat" not in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims and "lon" not in ds.dims:
        rename["longitude"] = "lon"
    return ds.rename(rename) if rename else ds


def subset_by_years(ds: xr.Dataset, years: Iterable[int]) -> xr.Dataset:
    """Return dataset times belonging to any of the provided years."""
    years = list(years)
    if not years:
        raise ValueError("At least one year must be provided.")
    mask = ds["time"].dt.year.isin(years)
    return ds.sel(time=mask)


def load_land_mask(
    mask_path: str | Path | None,
    *,
    variable: str,
) -> xr.DataArray | None:
    """Load a land mask DataArray with consistent latitude/longitude dims."""
    if mask_path is None:
        return None
    mask_path = Path(mask_path).expanduser().resolve()
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    mask_ds = xr.open_dataset(mask_path)
    if variable not in mask_ds:
        raise KeyError(f"{variable!r} not found in {mask_path}")
    da = ensure_lat_lon_dims(mask_ds[variable])
    return da
