"""
Lotus-2 Pseudo-Label Training Module

This module provides functionality to use Lotus-2 as a teacher model
for generating pseudo labels to train student models.
"""

from .teacher_model import Lotus2Teacher, load_lotus2_teacher
from .pseudo_label_dataset import PseudoLabelDataset, create_pseudo_label_dataset
from .train_utils import (
    PseudoLabelLoss,
    DepthConsistencyLoss,
    NormalConsistencyLoss,
    compute_depth_metrics,
    compute_normal_metrics,
)
from .config import PseudoLabelTrainingConfig

__all__ = [
    "Lotus2Teacher",
    "load_lotus2_teacher",
    "PseudoLabelDataset",
    "create_pseudo_label_dataset",
    "PseudoLabelLoss",
    "DepthConsistencyLoss",
    "NormalConsistencyLoss",
    "compute_depth_metrics",
    "compute_normal_metrics",
    "PseudoLabelTrainingConfig",
]
