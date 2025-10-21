# NeuralCORA

Utilities, dataset wrappers, and baseline models for working with the NeuralCORA
atmospheric forecast benchmark. The code was extracted from the original
quickstart notebook and reorganised into importable modules under `src/` so the
project is ready to publish on GitHub.

## What's Included

- `src/data.py` — helpers for locating yearly NetCDF files and opening a time
  window with `xarray`.
- `src/datasets.py` — a `CoraDataset` PyTorch dataset that prepares sliding
  windows and normalisation statistics.
- `src/models.py` — a lightweight convolutional baseline using periodic
  padding.
- `src/resnet.py` — a deeper periodic ResNet with residual blocks for
  improved baselines.
- `src/training.py` — training loop and batched inference helpers.
- `src/metrics.py` — scoring utilities (currently RMSE with optional latitude
  weights).
- `quickstart.ipynb` — refreshed to import the modules above instead of
  redefining everything inline.
- `quickstart-resnet.ipynb` — trains the periodic ResNet baseline end-to-end.

## Requirements

This repository assumes you already created an environment with the packages
used in the original notebook (`xarray`, `torch`, `pandas`, `numpy`,
`matplotlib`, `seaborn`, etc.). See `docs/configuration.md` for suggested
package versions and optional extras.

## Usage

1. Point the notebook (or your own scripts) at a folder containing yearly
   NeuralCORA NetCDF files named like `NY_2020_180_360.nc`.
2. Import the helpers from `src`:

```python
from src import (
    open_period,
    subset_by_years,
    load_land_mask,
    CoraDataset,
    NeuralCoraCNN,
    fit,
    create_predictions,
    rename_for_score,
    compute_weighted_rmse,
)
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

TIME_PERIOD = "2020-2022"
ds = open_period(TIME_PERIOD, data_dir="/path/to/data")
ds = ds[["zeta"]]

train_ds = CoraDataset(ds, OrderedDict({"zeta": None}), lead_time=1, input_steps=9)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

model = NeuralCoraCNN(train_ds.input_channels, [8, 32, 32, 8], kernel_size=3, out_channels=train_ds.output_channels)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = fit(model, train_loader, optimizer=optimizer, criterion=criterion, epochs=5, device="cpu")
```

3. Refer to `quickstart.ipynb` for a complete workflow that loads data,
   trains the CNN, and evaluates the forecast RMSE.

## Contributing

- Run the notebook (or your own scripts) from the project root so Python can
  resolve the `src` package.
- Format and linting are not enforced yet; keep imports clean and add concise
  comments only where behaviour is non-obvious.
- Open an issue or pull request if you would like to contribute additional
  models or diagnostics.
