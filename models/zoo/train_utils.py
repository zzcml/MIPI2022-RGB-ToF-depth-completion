"""
Training Utilities for RoboDepth Zoo Models

Provides loss functions, metrics, and training utilities for pseudo-label based training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class RoboDepthLoss(nn.Module):
    """
    Combined loss function for depth estimation with pseudo-labels.
    
    Supports multiple loss types:
    - MSE: Mean Squared Error
    - SILog: Scale-Invariant Log Loss
    - BerHu: Reverse Huber Loss
    - Combined: Weighted combination of multiple losses
    """
    
    def __init__(
        self,
        loss_type: str = "combined",
        mse_weight: float = 1.0,
        silog_weight: float = 0.5,
        berhu_weight: float = 0.5,
        threshold: float = 0.2,
        eps: float = 1e-8
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.mse_weight = mse_weight
        self.silog_weight = silog_weight
        self.berhu_weight = berhu_weight
        self.threshold = threshold
        self.eps = eps
        
        # Initialize individual loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the loss between prediction and target.
        
        Args:
            pred: Predicted depth maps (B, 1, H, W)
            target: Target depth maps (B, 1, H, W)
            mask: Valid pixel mask (B, 1, H, W) or (B, H, W)
            
        Returns:
            loss: Computed loss value
        """
        # Ensure same shape
        if pred.shape != target.shape:
            target = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.shape != pred.shape:
                mask = F.interpolate(mask, size=pred.shape[-2:], mode='nearest')
        else:
            # Create default mask (valid where both pred and target are positive and finite)
            mask = (pred > 0) & (target > 0) & torch.isfinite(pred) & torch.isfinite(target)
        
        # Ensure minimum valid pixels
        if mask.sum() < 10:
            mask = torch.ones_like(pred).bool()
        
        if self.loss_type == "mse":
            return self._mse_loss(pred, target, mask)
        elif self.loss_type == "silog":
            return self._silog_loss(pred, target, mask)
        elif self.loss_type == "berhu":
            return self._berhu_loss(pred, target, mask)
        elif self.loss_type == "combined":
            return self._combined_loss(pred, target, mask)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _mse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute masked MSE loss."""
        loss = self.mse_loss(pred, target)
        loss = loss * mask.float()
        return loss.sum() / (mask.sum() + self.eps)
    
    def _silog_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Scale-Invariant Log Loss.
        
        Reference: "Deep Ordinal Regression Network for Monocular Depth Estimation"
        """
        # Add epsilon to avoid log(0)
        pred_safe = torch.clamp(pred, min=self.eps)
        target_safe = torch.clamp(target, min=self.eps)
        
        # Log transform
        log_pred = torch.log(pred_safe)
        log_target = torch.log(target_safe)
        
        # Apply mask
        log_pred = log_pred[mask]
        log_target = log_target[mask]
        
        if len(log_pred) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Compute differences
        diff = log_pred - log_target
        
        # Scale-invariant loss
        n = len(diff)
        loss = (diff ** 2).mean() - (diff.mean() ** 2) * (n / (n - 1))
        
        return loss
    
    def _berhu_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Reverse Huber (BerHu) Loss.
        
        Reference: "Deep Ordinal Regression Network for Monocular Depth Estimation"
        """
        # Compute absolute error
        abs_error = torch.abs(pred - target)
        abs_error = abs_error[mask]
        
        if len(abs_error) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Find threshold
        c = self.threshold * abs_error.max()
        
        # BerHu loss
        berhu_mask = abs_error <= c
        loss = torch.where(
            berhu_mask,
            abs_error,
            (abs_error ** 2 + c ** 2) / (2 * c)
        )
        
        return loss.mean()
    
    def _combined_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted combination of losses."""
        total_loss = 0.0
        
        if self.mse_weight > 0:
            total_loss += self.mse_weight * self._mse_loss(pred, target, mask)
        
        if self.silog_weight > 0:
            total_loss += self.silog_weight * self._silog_loss(pred, target, mask)
        
        if self.berhu_weight > 0:
            total_loss += self.berhu_weight * self._berhu_loss(pred, target, mask)
        
        return total_loss


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    thresholds: List[float] = [1.25, 1.25**2, 1.25**3]
) -> Dict[str, float]:
    """
    Compute depth estimation metrics.
    
    Args:
        pred: Predicted depth maps
        target: Ground truth depth maps
        mask: Valid pixel mask
        thresholds: Thresholds for delta accuracy computation
        
    Returns:
        Dictionary containing:
            - abs_rel: Absolute Relative Error
            - sq_rel: Squared Relative Error
            - rmse: Root Mean Square Error
            - log_rmse: Log RMSE
            - delta1, delta2, delta3: Accuracy within threshold
    """
    if pred.shape != target.shape:
        target = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)
    
    if mask is None:
        mask = (pred > 0) & (target > 0) & torch.isfinite(pred) & torch.isfinite(target)
    else:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape != pred.shape:
            mask = F.interpolate(mask, size=pred.shape[-2:], mode='nearest')
    
    # Flatten for computation
    pred_flat = pred[mask].squeeze()
    target_flat = target[mask].squeeze()
    
    if len(pred_flat) == 0:
        return {
            "abs_rel": 0.0,
            "sq_rel": 0.0,
            "rmse": 0.0,
            "log_rmse": 0.0,
            "delta1": 0.0,
            "delta2": 0.0,
            "delta3": 0.0,
        }
    
    eps = 1e-8
    
    # Absolute Relative Error
    abs_rel = torch.abs(pred_flat - target_flat) / (target_flat + eps)
    abs_rel = abs_rel.mean().item()
    
    # Squared Relative Error
    sq_rel = ((pred_flat - target_flat) ** 2) / (target_flat + eps)
    sq_rel = sq_rel.mean().item()
    
    # RMSE
    rmse = torch.sqrt(((pred_flat - target_flat) ** 2).mean()).item()
    
    # Log RMSE
    log_pred = torch.log(torch.clamp(pred_flat, min=eps))
    log_target = torch.log(torch.clamp(target_flat, min=eps))
    log_rmse = torch.sqrt(((log_pred - log_target) ** 2).mean()).item()
    
    # Delta accuracy
    ratio = torch.max(pred_flat / (target_flat + eps), target_flat / (pred_flat + eps))
    deltas = {}
    for i, thresh in enumerate(thresholds):
        deltas[f"delta{i+1}"] = (ratio < thresh).float().mean().item()
    
    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "log_rmse": log_rmse,
        **deltas,
    }


def get_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    scheduler_type: str = "cosine"
):
    """
    Create learning rate scheduler with warmup.
    
    Args:
        optimizer: Optimizer instance
        max_epochs: Maximum number of epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
        scheduler_type: Type of scheduler ('cosine', 'step', 'linear')
        
    Returns:
        Scheduler instance
    """
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        StepLR,
        LambdaLR,
        SequentialLR
    )
    
    # Warmup scheduler
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # Main scheduler
    if scheduler_type == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=min_lr
        )
    elif scheduler_type == "step":
        main_scheduler = StepLR(
            optimizer,
            step_size=max_epochs // 3,
            gamma=0.1
        )
    elif scheduler_type == "linear":
        def linear_lambda(epoch):
            if epoch < warmup_epochs:
                return 1.0
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            return max(min_lr, 1.0 - progress)
        main_scheduler = LambdaLR(optimizer, lr_lambda=linear_lambda)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Combine warmup and main scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    return scheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: str,
    filename: str = "checkpoint.pth.tar",
    is_best: bool = False
) -> str:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state (optional)
        epoch: Current epoch
        metrics: Training metrics
        output_dir: Output directory
        filename: Checkpoint filename
        is_best: Whether this is the best checkpoint
        
    Returns:
        Path to saved checkpoint
    """
    import os
    import shutil
    
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(output_dir, "checkpoint_best.pth.tar")
        shutil.copyfile(checkpoint_path, best_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional = None,
    device: str = "cuda"
) -> Tuple[int, Dict[str, float]]:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state (optional)
        scheduler: Scheduler to load state (optional)
        device: Device to load to
        
    Returns:
        Tuple of (epoch, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    return epoch, metrics


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
        return f"{self.name}: {self.avg:.4f}"
