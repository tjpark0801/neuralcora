"""NeuralCORA library components extracted from the quickstart notebook."""

from .data import collect_year_files, ensure_lat_lon_dims, load_land_mask, open_period, parse_period, subset_by_years
from .datasets import CoraDataset, VariableConfig
from .metrics import compute_weighted_rmse, rename_for_score
from .models import NeuralCoraCNN, PeriodicConvBlock
from .resnet import NeuralCoraResNet, PeriodicConv2d, ResidualBlock
from .training import create_predictions, fit

__all__ = [
    "collect_year_files",
    "ensure_lat_lon_dims",
    "load_land_mask",
    "open_period",
    "parse_period",
    "subset_by_years",
    "CoraDataset",
    "VariableConfig",
    "compute_weighted_rmse",
    "rename_for_score",
    "NeuralCoraCNN",
    "NeuralCoraResNet",
    "PeriodicConvBlock",
    "PeriodicConv2d",
    "ResidualBlock",
    "create_predictions",
    "fit",
]
