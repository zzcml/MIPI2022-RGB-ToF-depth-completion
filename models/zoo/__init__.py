"""
RoboDepth Zoo Model Training Module

This module provides functionality to use models from RoboDepth zoo
as selectable training models (teacher or student) for pseudo-label training.

Reference: https://github.com/worldbench/RoboDepth
"""

from .config import RoboDepthZooConfig, MODEL_ZOO_REGISTRY
from .model_wrapper import RoboDepthZooWrapper
from .pseudo_label_dataset import RoboDepthPseudoLabelDataset
from .train_utils import (
    RoboDepthLoss,
    compute_depth_metrics,
    get_learning_rate_scheduler,
    save_checkpoint,
    load_checkpoint
)
from .train_pseudo_label import train_robodepth_zoo

__all__ = [
    "RoboDepthZooConfig",
    "MODEL_ZOO_REGISTRY",
    "RoboDepthZooWrapper",
    "RoboDepthPseudoLabelDataset",
    "RoboDepthLoss",
    "compute_depth_metrics",
    "get_learning_rate_scheduler",
    "save_checkpoint",
    "load_checkpoint",
    "train_robodepth_zoo"
]
