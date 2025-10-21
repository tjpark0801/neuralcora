# NeuralCORA Technical Guide

## 1. Project Overview

NeuralCORA packages the data tooling, baseline models, and evaluation workflow
used in the NeuralCORA atmospheric forecasting benchmark. The original code
lived inside a monolithic notebook; this repository restructures it into an
importable Python package (`src/`) while keeping an opinionated quickstart
notebook for exploratory runs.

All modelling code now targets **PyTorch**. Legacy TensorFlow builders were
ported to torch-native modules so you can reuse familiar configuration knobs
(`build_resnet`, `build_unet`, latitude-weighted losses, etc.) without juggling
frameworks.

## 2. Repository Layout

| Path | Purpose |
|------|---------|
| `src/data.py` | Locate yearly NetCDF files, open date ranges, apply land masks. |
| `src/datasets.py` | `CoraDataset` that prepares sliding windows, normalises features, and exposes PyTorch tensors. |
| `src/networks.py` | Torch implementations of the periodic CNN, ResNet, U-ResNet, UNet variants, and latitude-aware losses. |
| `src/cnn.py` / `src/models.py` | Backward-compatible exports for the simple periodic CNN. |
| `src/resnet.py` | Re-exports the residual stack from `networks`. |
| `src/training.py` | Mini training loop (`fit`) and batched inference helper (`create_predictions`). |
| `src/metrics.py` | RMSE and renaming helpers; latitude weighting utilises `xarray`. |
| `quickstart.ipynb` | End-to-end tutorial: data prep → training → interactive diagnostics. |
| `docs/` | Supplementary documentation (this guide, configuration tips). |

## 3. Getting Started

### 3.1 Dependencies

Create a Python environment (conda or venv) with the following packages:

- `python>=3.10`
- `pytorch` (CPU or CUDA build)
- `numpy`, `pandas`, `xarray`, `netcdf4`
- `matplotlib`, `seaborn`, `plotly`
- `ipywidgets`
- `tqdm` (optional progress bars)

Reference `docs/configuration.md` for a ready-to-copy conda environment.

### 3.2 Data Expectations

NeuralCORA stores hourly atmospheric fields per year (e.g.
`NY_2020_180_360.nc`). Place the NetCDF files in a directory accessible to the
notebook or scripts. Optional helpers can apply a land/sea mask stored in a
separate NetCDF file (`real_land_mask_180_360.nc`).

## 4. Typical Workflow

1. **Open a multi-year window**

   ```python
   from src import open_period, ensure_lat_lon_dims
   ds = open_period("2020-2022", data_dir="/data/neuralcora")
   ds = ensure_lat_lon_dims(ds)[["zeta"]]  # keep the sea-level variable
   ```

2. **Split by year**

   ```python
   from src import subset_by_years
   train_ds = subset_by_years(ds, [2020, 2021])
   valid_ds = subset_by_years(ds, [2022])
   ```

3. **Instantiate datasets & loaders**

   ```python
   from collections import OrderedDict
   from src import CoraDataset
   from torch.utils.data import DataLoader

   var_dict = OrderedDict({"zeta": None})
   train = CoraDataset(train_ds, var_dict, lead_time=1, input_steps=4)
   valid = CoraDataset(valid_ds, var_dict, lead_time=1, input_steps=4,
                       mean=train.mean, std=train.std)

   train_loader = DataLoader(train, batch_size=8, shuffle=True)
   valid_loader = DataLoader(valid, batch_size=8)
   ```

4. **Pick a model**

   ```python
   from src.networks import build_resnet

   model = build_resnet(
       filters=[32, 32, 32, train.output_channels],
       kernels=[3, 3, 3, 3],
       input_shape=(*train.inputs.shape[-2:], train.input_channels),
       bn_position="post",
       dropout=0.1,
   ).to(device)
   ```

   If you prefer the CNN baseline, use `NeuralCoraCNN`. For UNet/U-ResNet
   architectures, call `build_unet` / `build_uresnet`.

5. **Train & monitor**

   ```python
   import torch
   from src import fit

   criterion = torch.nn.MSELoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

   history = fit(
       model,
       train_loader,
       optimizer=optimizer,
       criterion=criterion,
       epochs=10,
       device=device,
       valid_loader=valid_loader,
   )
   ```

6. **Evaluate / visualise**

   ```python
   from src import create_predictions, rename_for_score, compute_weighted_rmse

   preds = create_predictions(model, valid_loader, valid, device)
   truth = rename_for_score(valid.ds["zeta"].sel(time=valid.valid_time))
   score = compute_weighted_rmse(rename_for_score(preds["zeta"]), truth)
   print("Weighted RMSE:", score.values.item())
   ```

7. **Inspect interactively (optional)**

   Run the final cells in `quickstart.ipynb` to launch an interactive Plotly
   widget. You can pass explicit latitude/longitude pairs via
   `comparison_coords` to focus on specific coastal stations.

## 5. Model Zoo

All model builders live in `src/networks.py`. They share periodic padding along
longitude and include options mirroring the original TensorFlow scripts.

| Builder | Highlights |
|---------|------------|
| `NeuralCoraCNN` | Stacked periodic conv blocks with ReLU & optional dropout. |
| `NeuralCoraResNet` / `build_resnet` | Residual blocks with configurable BN placement, skip connections, categorical outputs. |
| `build_uresnet` | U-shaped residual encoder/decoder with skip concatenations. |
| `build_unet` | Residual UNet inspired by ResNet34 UNet variants. |
| `build_unet_google` | Agrawal et al. flavour UNet with periodic blocks. |

All builders accept `input_shape=(lat, lon, channels)` and return PyTorch
`nn.Module` instances ready for training.

### Latitude-Aware Losses

Functions like `create_lat_mse`, `create_lat_rmse`, and CRPS variants generate
callables compatible with `torch.nn.Module` training loops. They compute cosine
latitude weights internally—pass your latitude array once and reuse the
returned closure.

## 6. Quickstart Notebook Highlights

- **Configuration cell** lets you define `comparison_coords`, `num_points`, and
  `time_window`. The latter trims prediction windows before ranking high-error
  sites.
- **Interactive Plotly widget** consumes either the top-RMSE sites or the exact
  coordinates you provide. Hover for tooltips, zoom, or export as PNG.
- **Model switch** (`model_kind`) toggles between CNN, ResNet, U-ResNet, UNet,
  and Google UNet. The selection is routed to the torch builders described
  above.

Running the notebook end-to-end is the quickest way to validate new datasets
or sanity-check model tweaks.

## 7. Extending the Repository

- **Add a model**: Implement it in `src/networks.py` (or a new module), expose
  it via `__all__`, and register it in the quickstart `model_factory` so users
  can select it interactively.
- **New metrics**: Use `xarray` for spatial weighting and expose them from
  `src/metrics.py`. Remember to add unit tests or notebook snippets proving
  correctness.
- **Data augmentations**: Extend `CoraDataset` or write wrapper datasets that
  perform on-the-fly transforms before batching.

## 8. Troubleshooting & Tips

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Notebook import errors (`ModuleNotFoundError: src`) | Notebook not executed from repo root. | Start Jupyter in the project root or add the path to `sys.path`. |
| Training diverges quickly | Learning rate too high or dropout disabled. | Lower LR (`1e-3 → 5e-4`) or enable `dropout`. |
| `torch` not found when running scripts | PyTorch missing from environment. | Install CPU build (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). |
| Interactive widget fails with `ValueError: No locations available` | `comparison_coords` empty and `selected_sites` not computed (preceding cell skipped). | Re-run the "Compare predictions and truth" cell before the interactive one. |

## 9. Support & Contributions

- File issues for bugs, enhancement requests, or documentation gaps.
- For substantial code changes, open a pull request with:
  - Updated notebook output (run clean).
  - Notes about performance or behavioural differences.
  - Additional docs/tests where relevant.

Happy forecasting!
