# NeuralCORA Configuration Guide

This page summarises the minimum configuration needed to run the quickstart
notebook or build experiments with the `src` modules.

## Python Environment

- Python 3.10 or later (the notebook was developed with 3.11).
- Core packages: `xarray`, `netcdf4`, `numpy`, `pandas`, `matplotlib`,
  `seaborn`, `torch`, `torchvision` (optional for utilities), and `tqdm`.
- Optional packages: `dask` for chunked loading, `cartopy` for nicer maps,
  `torchmetrics` for extended evaluation.

You can build an environment with `conda`:

```bash
conda create -n neuralcora python=3.11 xarray netcdf4 numpy pandas matplotlib seaborn pytorch cpuonly -c pytorch -c conda-forge
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # create this file for your project if desired
```

## Data Directory

Download the yearly NeuralCORA NetCDF files and place them in a directory, for
example `/data/neuralcora`. Files should follow the naming pattern
`NY_<year>_180_360.nc`.

Update the `data_dir` variable at the top of `quickstart.ipynb` (or pass the
path directly to `open_period`) so the helpers can locate the files.

## Land Mask (Optional)

If you have a land mask file such as `real_land_mask_180_360.nc`, set
`MASK_PATH` to point to it. The helper `load_land_mask` takes care of loading
and aligning the mask with the forecast variable.

## Device Selection

The notebook automatically selects CUDA, TPU (via `torch_xla`), Apple MPS, or
CPU in that order. You can override this by setting `device = torch.device("cpu")`
or similar before calling `fit`.
