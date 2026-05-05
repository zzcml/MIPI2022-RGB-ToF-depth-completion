"""
Depth Anything V2 Pseudo-Label Training Module

This module provides functionality to use Depth Anything V2 (ViT-L) as a teacher model
to generate pseudo-labels for training student models (ViT-S, ViT-B, or custom models).

Based on: https://github.com/DepthAnything/Depth-Anything-V2
"""

from .config import DA2TrainingConfig
from .teacher_model import DepthAnythingV2Teacher
from .student_model import StudentModelWrapper
from .pseudo_label_dataset import DA2PseudoLabelDataset
from .train_utils import (
    PseudoLabelLoss,
    compute_depth_metrics,
    validate_student_model,
)
from .pipeline_wrapper import DA2PipelineWrapper

__all__ = [
    "DA2TrainingConfig",
    "DepthAnythingV2Teacher",
    "StudentModelWrapper",
    "DA2PseudoLabelDataset",
    "PseudoLabelLoss",
    "compute_depth_metrics",
    "validate_student_model",
    "DA2PipelineWrapper",
]
