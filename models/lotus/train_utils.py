"""
Training utilities for pseudo-label training with Lotus-2 as teacher.

This module provides:
1. Loss functions for pseudo-label training
2. Confidence-weighted loss functions
3. Evaluation metrics for depth and normal estimation
4. Training loop utilities
"""

import logging
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PseudoLabelLoss(nn.Module):
    """
    Loss function for pseudo-label training.
    
    Supports multiple loss types and optional confidence weighting.
    
    Args:
        loss_type: Type of loss ("l1", "l2", "smooth_l1")
        use_confidence: Whether to use confidence weighting
        confidence_threshold: Minimum confidence threshold for filtering
    """
    
    def __init__(
        self,
        loss_type: str = "l1",
        use_confidence: bool = True,
        confidence_threshold: float = 0.0,
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.use_confidence = use_confidence
        self.confidence_threshold = confidence_threshold
        
        if loss_type == "l1":
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == "l2":
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == "smooth_l1":
            self.criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        prediction: torch.Tensor,
        pseudo_label: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the pseudo-label loss.
        
        Args:
            prediction: Model prediction
            pseudo_label: Pseudo label from teacher model
            confidence: Optional confidence scores for weighting
            mask: Optional mask to filter valid pixels
        
        Returns:
            Computed loss value
        """
        # Ensure same shape
        if prediction.shape != pseudo_label.shape:
            # Try to broadcast or squeeze/unsqueeze
            if prediction.dim() > pseudo_label.dim():
                if prediction.shape[1] == 1 and pseudo_label.dim() == 3:
                    prediction = prediction.squeeze(1)
                elif pseudo_label.dim() == 2:
                    pseudo_label = pseudo_label.unsqueeze(0)
            elif prediction.dim() < pseudo_label.dim():
                if pseudo_label.shape[0] == 1 and prediction.dim() == 3:
                    pseudo_label = pseudo_label.squeeze(0)
        
        # Compute base loss
        loss = self.criterion(prediction, pseudo_label)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
        
        # Apply confidence weighting if provided
        if self.use_confidence and confidence is not None:
            # Ensure confidence has same shape as loss
            if confidence.dim() < loss.dim():
                while confidence.dim() < loss.dim():
                    confidence = confidence.unsqueeze(-1)
            
            # Filter by confidence threshold
            confidence_mask = (confidence > self.confidence_threshold).float()
            confidence_weight = confidence * confidence_mask
            
            loss = loss * confidence_weight
            
            # Normalize by sum of weights
            weight_sum = confidence_weight.sum()
            if weight_sum > 0:
                loss = loss.sum() / weight_sum
            else:
                loss = loss.mean()
        else:
            loss = loss.mean()
        
        return loss


class DepthConsistencyLoss(nn.Module):
    """
    Depth consistency loss for multi-view or temporal consistency.
    
    This loss encourages consistent depth predictions across related images.
    """
    
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == "l1":
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == "l2":
            self.criterion = nn.MSELoss(reduction='none')
        else:
            self.criterion = nn.L1Loss(reduction='none')
    
    def forward(
        self,
        pred_depth_1: torch.Tensor,
        pred_depth_2: torch.Tensor,
        flow_field: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute depth consistency loss between two predictions.
        
        Args:
            pred_depth_1: Depth prediction for first view
            pred_depth_2: Depth prediction for second view
            flow_field: Optional optical flow to warp predictions
            mask: Optional mask for valid regions
        
        Returns:
            Consistency loss value
        """
        if flow_field is not None:
            # Warp pred_depth_2 to align with pred_depth_1
            pred_depth_2_warped = self._warp_with_flow(pred_depth_2, flow_field)
        else:
            pred_depth_2_warped = pred_depth_2
        
        loss = self.criterion(pred_depth_1, pred_depth_2_warped)
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum()
        else:
            return loss.mean()
    
    def _warp_with_flow(
        self, 
        depth: torch.Tensor, 
        flow: torch.Tensor
    ) -> torch.Tensor:
        """Warp depth map using optical flow."""
        # Simplified warping - in practice, you'd use grid_sample
        # This is a placeholder implementation
        return depth


class NormalConsistencyLoss(nn.Module):
    """
    Normal map consistency loss.
    
    Uses cosine similarity to measure alignment between normal vectors.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        pred_normal: torch.Tensor,
        pseudo_normal: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute normal consistency loss using cosine similarity.
        
        Args:
            pred_normal: Predicted normal maps (B, 3, H, W)
            pseudo_normal: Pseudo label normal maps (B, 3, H, W)
            mask: Optional mask for valid pixels
        
        Returns:
            Normal consistency loss
        """
        # Normalize both predictions
        pred_normal = F.normalize(pred_normal, dim=1)
        pseudo_normal = F.normalize(pseudo_normal, dim=1)
        
        # Cosine similarity (higher is better, so we negate for loss)
        cos_sim = (pred_normal * pseudo_normal).sum(dim=1)
        
        # Convert to loss (1 - cos_sim ranges from 0 to 2)
        loss = 1 - cos_sim
        
        if mask is not None:
            loss = loss * mask.squeeze(1)
            return loss.sum() / mask.sum()
        else:
            return loss.mean()


def compute_depth_metrics(
    prediction: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Compute depth estimation metrics.
    
    Args:
        prediction: Predicted depth map
        target: Ground truth or pseudo-label depth map
        mask: Optional mask for valid pixels
    
    Returns:
        Dictionary of metrics (abs_rel, sq_rel, rmse, log_rmse, delta ratios)
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Ensure 2D arrays
    if prediction.ndim == 3:
        prediction = prediction.mean(axis=-1)
    if target.ndim == 3:
        target = target.mean(axis=-1)
    
    # Apply mask
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.mean(axis=-1)
        prediction = prediction[mask > 0]
        target = target[mask > 0]
    else:
        prediction = prediction.flatten()
        target = target.flatten()
    
    # Remove invalid values
    valid_mask = (target > 0) & (~np.isnan(prediction)) & (~np.isinf(prediction))
    prediction = prediction[valid_mask]
    target = target[valid_mask]
    
    if len(prediction) == 0:
        return {
            "abs_rel": float('nan'),
            "sq_rel": float('nan'),
            "rmse": float('nan'),
            "log_rmse": float('nan'),
            "delta1": float('nan'),
            "delta2": float('nan'),
            "delta3": float('nan'),
        }
    
    # Scale-invariant evaluation
    prediction = prediction + np.log(np.median(target) / np.median(prediction))
    
    # Metrics
    diff = prediction - target
    
    abs_rel = np.mean(np.abs(diff) / target)
    sq_rel = np.mean(diff ** 2 / target)
    rmse = np.sqrt(np.mean(diff ** 2))
    log_rmse = np.sqrt(np.mean(diff ** 2))
    
    # Delta ratios
    thresh = np.maximum(target / prediction, prediction / target)
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25 ** 2).mean()
    delta3 = (thresh < 1.25 ** 3).mean()
    
    return {
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "log_rmse": float(log_rmse),
        "delta1": float(delta1),
        "delta2": float(delta2),
        "delta3": float(delta3),
    }


def compute_normal_metrics(
    prediction: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Compute normal estimation metrics.
    
    Args:
        prediction: Predicted normal maps
        target: Ground truth or pseudo-label normal maps
        mask: Optional mask for valid pixels
    
    Returns:
        Dictionary of metrics (mean_error, median_error, accuracy thresholds)
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Ensure correct shape (B, 3, H, W) or (3, H, W)
    if prediction.ndim == 4:
        prediction = prediction[0]
    if target.ndim == 4:
        target = target[0]
    
    # Transpose to (H, W, 3) if needed
    if prediction.shape[0] == 3:
        prediction = np.transpose(prediction, (1, 2, 0))
    if target.shape[0] == 3:
        target = np.transpose(target, (1, 2, 0))
    
    # Apply mask
    if mask is not None:
        if mask.ndim == 4:
            mask = mask[0]
        if mask.shape[0] == 1 or mask.shape[0] == 3:
            mask = np.transpose(mask, (1, 2, 0))
        prediction = prediction[mask > 0]
        target = target[mask > 0]
    else:
        prediction = prediction.reshape(-1, 3)
        target = target.reshape(-1, 3)
    
    # Remove invalid values
    valid_mask = (~np.isnan(prediction).any(axis=1)) & (~np.isnan(target).any(axis=1))
    prediction = prediction[valid_mask]
    target = target[valid_mask]
    
    if len(prediction) == 0:
        return {
            "mean_error": float('nan'),
            "median_error": float('nan'),
            "acc_11.25": float('nan'),
            "acc_22.5": float('nan'),
            "acc_30": float('nan'),
        }
    
    # Normalize
    prediction = prediction / (np.linalg.norm(prediction, axis=1, keepdims=True) + 1e-8)
    target = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    
    # Compute angle error
    dot_product = np.sum(prediction * target, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_error = np.arccos(dot_product) * 180.0 / np.pi
    
    # Metrics
    mean_error = np.mean(angle_error)
    median_error = np.median(angle_error)
    acc_11_25 = (angle_error < 11.25).mean()
    acc_22_5 = (angle_error < 22.5).mean()
    acc_30 = (angle_error < 30.0).mean()
    
    return {
        "mean_error": float(mean_error),
        "median_error": float(median_error),
        "acc_11.25": float(acc_11_25),
        "acc_22.5": float(acc_22_5),
        "acc_30": float(acc_30),
    }


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.6f}"


def get_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.0,
    optimizer_type: str = "adamw",
) -> torch.optim.Optimizer:
    """
    Create an optimizer for training.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        optimizer_type: Type of optimizer ("adam", "adamw", "sgd")
    
    Returns:
        Optimizer instance
    """
    if optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_type: str = "cosine",
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        scheduler_type: Type of scheduler ("cosine", "linear", "constant")
    
    Returns:
        Scheduler instance
    """
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=0,
        )
    elif scheduler_type == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=num_training_steps,
        )
    elif scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
