"""
Training utilities for Depth Anything V2 pseudo-label training.

This module provides loss functions, evaluation metrics, and other utilities
for training student models with pseudo-labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import numpy as np


class PseudoLabelLoss(nn.Module):
    """
    Loss function for pseudo-label training.
    
    Supports multiple loss types:
    - MSE: Mean Squared Error
    - SILog: Scale-Invariant Log Loss
    - Combined: Weighted combination of multiple losses
    """
    
    def __init__(
        self,
        loss_type: str = "combined",
        confidence_weighting: bool = True,
        alpha: float = 0.5,  # Weight for MSE in combined loss
        beta: float = 0.5,   # Weight for SILog in combined loss
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.confidence_weighting = confidence_weighting
        self.alpha = alpha
        self.beta = beta
        
        if loss_type == "combined":
            self.mse_loss = nn.MSELoss(reduction='none')
            self.silog_loss = ScaleInvariantLogLoss()
        elif loss_type == "mse":
            self.mse_loss = nn.MSELoss(reduction='none')
        elif loss_type == "silog":
            self.silog_loss = ScaleInvariantLogLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        confidence_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets.
        
        Args:
            predictions: Predicted depth maps (B, H, W)
            targets: Target depth maps (pseudo-labels) (B, H, W)
            confidence_weights: Optional confidence weights (B, H, W)
            
        Returns:
            Scalar loss value
        """
        # Ensure same shape
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        if confidence_weights is not None and self.confidence_weighting:
            if confidence_weights.dim() == 3:
                confidence_weights = confidence_weights.unsqueeze(1)
        else:
            confidence_weights = torch.ones_like(targets)
        
        if self.loss_type == "combined":
            mse_loss = self.mse_loss(predictions, targets)
            silog_loss = self.silog_loss(predictions, targets)
            
            # Apply confidence weighting
            mse_loss = (mse_loss * confidence_weights).sum() / confidence_weights.sum()
            silog_loss = (silog_loss * confidence_weights).sum() / confidence_weights.sum()
            
            loss = self.alpha * mse_loss + self.beta * silog_loss
            
        elif self.loss_type == "mse":
            loss = self.mse_loss(predictions, targets)
            loss = (loss * confidence_weights).sum() / confidence_weights.sum()
            
        elif self.loss_type == "silog":
            loss = self.silog_loss(predictions, targets)
            loss = (loss * confidence_weights).sum() / confidence_weights.sum()
        
        return loss


class ScaleInvariantLogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss (SILog).
    
    This loss is commonly used for depth estimation as it is invariant
    to global scale changes.
    
    Reference: "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
    """
    
    def __init__(self, alpha: float = 0.5, max_depth: float = 10.0):
        super().__init__()
        self.alpha = alpha
        self.max_depth = max_depth
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SILog loss.
        
        Args:
            predictions: Predicted depth (B, C, H, W)
            targets: Target depth (B, C, H, W)
            
        Returns:
            SILog loss value
        """
        # Clip predictions and targets to avoid log(0)
        predictions = torch.clamp(predictions, min=1e-8, max=self.max_depth)
        targets = torch.clamp(targets, min=1e-8, max=self.max_depth)
        
        # Log transform
        log_pred = torch.log(predictions)
        log_target = torch.log(targets)
        
        # Compute differences
        diff = log_pred - log_target
        
        # SILog loss: mean of squared differences - alpha * (mean of differences)^2
        loss = diff.pow(2).mean(dim=(1, 2, 3)) - self.alpha * diff.mean(dim=(1, 2, 3)).pow(2)
        
        return loss.mean()


def compute_depth_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    thresholds: List[float] = [1.25, 1.25**2, 1.25**3],
) -> Dict[str, float]:
    """
    Compute depth estimation metrics.
    
    Args:
        predictions: Predicted depth maps
        targets: Ground truth / pseudo-label depth maps
        thresholds: Thresholds for delta accuracy computation
        
    Returns:
        Dictionary containing:
            - abs_rel: Absolute Relative Error
            - sq_rel: Squared Relative Error
            - rmse: Root Mean Squared Error
            - log_rmse: Log RMSE
            - delta_<threshold>: Percentage of pixels within threshold
    """
    # Ensure 4D tensors
    if predictions.dim() == 3:
        predictions = predictions.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    
    # Flatten for easier computation
    B = predictions.shape[0]
    pred_flat = predictions.view(B, -1)
    target_flat = targets.view(B, -1)
    
    # Mask for valid pixels (exclude zeros and very small values)
    valid_mask = target_flat > 1e-8
    
    metrics = {}
    
    # Absolute Relative Error
    abs_rel = ((pred_flat - target_flat).abs() / target_flat)[valid_mask].mean()
    metrics['abs_rel'] = abs_rel.item()
    
    # Squared Relative Error
    sq_rel = (((pred_flat - target_flat) ** 2) / target_flat)[valid_mask].mean()
    metrics['sq_rel'] = sq_rel.item()
    
    # RMSE
    rmse = ((pred_flat - target_flat) ** 2)[valid_mask].mean().sqrt()
    metrics['rmse'] = rmse.item()
    
    # Log RMSE
    log_pred = torch.log(torch.clamp(pred_flat, min=1e-8))
    log_target = torch.log(torch.clamp(target_flat, min=1e-8))
    log_rmse = ((log_pred - log_target) ** 2)[valid_mask].mean().sqrt()
    metrics['log_rmse'] = log_rmse.item()
    
    # Delta accuracy (percentage within threshold)
    ratio = torch.max(pred_flat / target_flat, target_flat / pred_flat)[valid_mask]
    for i, thresh in enumerate(thresholds):
        delta_acc = (ratio < thresh).float().mean()
        metrics[f'delta_{thresh:.2f}'] = delta_acc.item()
    
    return metrics


def validate_student_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Validate student model on a dataset.
    
    Args:
        model: Student model to validate
        dataloader: Validation data loader
        device: Device to run validation on
        loss_fn: Optional loss function to compute validation loss
        
    Returns:
        Dictionary containing all computed metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            depths = batch['depths'].to(device)
            confidences = batch.get('confidences', None)
            if confidences is not None:
                confidences = confidences.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss if provided
            if loss_fn is not None:
                loss = loss_fn(predictions, depths, confidences)
                total_loss += loss.item()
                num_batches += 1
            
            all_predictions.append(predictions.cpu())
            all_targets.append(depths.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_depth_metrics(all_predictions, all_targets)
    
    # Add average loss
    if loss_fn is not None and num_batches > 0:
        metrics['val_loss'] = total_loss / num_batches
    
    model.train()
    
    return metrics


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine decay.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate for current epoch."""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.current_epoch == 0:
            return 0.0
        
        if self.current_epoch <= self.warmup_epochs:
            return self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self, name: str = ''):
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
        return f'{self.name}: {self.avg:.4f}'
