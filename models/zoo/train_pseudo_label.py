"""
Training Script for RoboDepth Zoo Models

Complete training pipeline for pseudo-label based depth estimation using
models from the RoboDepth zoo.

Usage:
    python train_pseudo_label.py --teacher_model robodepth_dinov2_large \\
                                 --student_model robodepth_resnet50 \\
                                 --data_root /path/to/data \\
                                 --output_dir /path/to/output
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, Optional
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train depth estimation model using RoboDepth zoo models"
    )
    
    # Model selection
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="robodepth_dinov2_large",
        help="Teacher model name from MODEL_ZOO_REGISTRY"
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="robodepth_resnet50",
        help="Student model name from MODEL_ZOO_REGISTRY"
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights"
    )
    
    # Data configuration
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for dataset"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs="+",
        default=[448, 448],
        help="Input image size (single int or H W)"
    )
    parser.add_argument(
        "--pseudo_label_mode",
        type=str,
        choices=["precomputed", "online"],
        default="online",
        help="Pseudo-label generation mode"
    )
    parser.add_argument(
        "--precomputed_labels_dir",
        type=str,
        default=None,
        help="Directory containing precomputed pseudo-labels"
    )
    
    # Training configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of warmup epochs"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["mse", "silog", "berhu", "combined"],
        default="combined",
        help="Loss function type"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for pseudo-labels"
    )
    
    # Logging and checkpointing
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/robodepth_zoo",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Interval for logging training metrics"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Interval for saving checkpoints"
    )
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        default=True,
        help="Use TensorBoard logging"
    )
    
    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=True,
        help="Use mixed precision training"
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> 'RoboDepthZooConfig':
    """Create configuration from parsed arguments."""
    from .config import RoboDepthZooConfig
    
    # Process image size
    if len(args.image_size) == 1:
        image_size = (args.image_size[0], args.image_size[0])
    else:
        image_size = tuple(args.image_size[:2])
    
    config = RoboDepthZooConfig(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        use_pretrained=args.use_pretrained,
        data_root=args.data_root,
        image_size=image_size,
        pseudo_label_mode=args.pseudo_label_mode,
        precomputed_labels_dir=args.precomputed_labels_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        loss_type=args.loss_type,
        confidence_threshold=args.confidence_threshold,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        use_tensorboard=args.use_tensorboard,
        device=args.device,
        mixed_precision=args.mixed_precision,
    )
    
    return config


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional,
    epoch: int,
    config,
    writer=None,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = "cuda"
) -> Dict[str, float]:
    """Train for one epoch."""
    from .train_utils import AverageMeter, compute_depth_metrics
    
    model.train()
    
    # Meters for tracking metrics
    loss_meter = AverageMeter("Loss")
    time_meter = AverageMeter("Time")
    
    end = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        pseudo_labels = batch["pseudo_labels"].to(device)
        valid_masks = batch["valid_masks"].to(device)
        
        # Measure data loading time
        data_time = time.time() - end
        
        # Forward pass
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                predictions = model(images)
                loss = criterion(predictions, pseudo_labels, valid_masks)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            predictions = model(images)
            loss = criterion(predictions, pseudo_labels, valid_masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Compute metrics
        metrics = compute_depth_metrics(predictions, pseudo_labels, valid_masks)
        
        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        time_meter.update(time.time() - end)
        
        # Log progress
        if (batch_idx + 1) % config.log_interval == 0:
            log_msg = (
                f"Epoch: [{epoch+1}][{batch_idx+1}/{len(dataloader)}] | "
                f"Time: {time_meter.avg:.3f}s | "
                f"Loss: {loss_meter.avg:.4f} | "
                f"AbsRel: {metrics['abs_rel']:.4f} | "
                f"Delta1: {metrics['delta1']:.4f}"
            )
            print(log_msg)
            
            # TensorBoard logging
            if writer is not None:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar("train/loss", loss_meter.avg, global_step)
                writer.add_scalar("train/abs_rel", metrics["abs_rel"], global_step)
                writer.add_scalar("train/delta1", metrics["delta1"], global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
        
        end = time.time()
    
    return {
        "loss": loss_meter.avg,
        "abs_rel": metrics["abs_rel"],
        "sq_rel": metrics["sq_rel"],
        "rmse": metrics["rmse"],
        "log_rmse": metrics["log_rmse"],
        "delta1": metrics["delta1"],
        "delta2": metrics["delta2"],
        "delta3": metrics["delta3"],
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    config,
    writer=None,
    device: str = "cuda"
) -> Dict[str, float]:
    """Validate the model."""
    from .train_utils import AverageMeter, compute_depth_metrics
    
    model.eval()
    
    loss_meter = AverageMeter("Loss")
    metrics_accumulator = {}
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        pseudo_labels = batch["pseudo_labels"].to(device)
        valid_masks = batch["valid_masks"].to(device)
        
        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, pseudo_labels, valid_masks)
        
        # Update loss meter
        loss_meter.update(loss.item(), images.size(0))
        
        # Compute and accumulate metrics
        batch_metrics = compute_depth_metrics(predictions, pseudo_labels, valid_masks)
        for key, value in batch_metrics.items():
            if key not in metrics_accumulator:
                metrics_accumulator[key] = 0.0
            metrics_accumulator[key] += value * images.size(0)
    
    # Average metrics
    total_samples = len(dataloader.dataset)
    for key in metrics_accumulator:
        metrics_accumulator[key] /= total_samples
    
    metrics = {
        "loss": loss_meter.avg,
        **metrics_accumulator,
    }
    
    # Log results
    print(f"Validation - Epoch {epoch+1}: Loss={metrics['loss']:.4f}, "
          f"AbsRel={metrics['abs_rel']:.4f}, Delta1={metrics['delta1']:.4f}")
    
    # TensorBoard logging
    if writer is not None:
        global_step = epoch * len(dataloader)
        writer.add_scalar("val/loss", metrics["loss"], global_step)
        writer.add_scalar("val/abs_rel", metrics["abs_rel"], global_step)
        writer.add_scalar("val/delta1", metrics["delta1"], global_step)
    
    return metrics


def train_robodepth_zoo(config) -> None:
    """
    Main training function for RoboDepth zoo models.
    
    Args:
        config: RoboDepthZooConfig instance
    """
    from .model_wrapper import RoboDepthZooWrapper
    from .pseudo_label_dataset import RoboDepthPseudoLabelDataset, create_dataloader
    from .train_utils import (
        RoboDepthLoss,
        get_learning_rate_scheduler,
        save_checkpoint,
        AverageMeter,
    )
    
    # Print configuration
    print("=" * 60)
    print("RoboDepth Zoo Model Training")
    print("=" * 60)
    print(f"Teacher Model: {config.teacher_model}")
    print(f"Student Model: {config.student_model}")
    print(f"Image Size: {config.image_size}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Output Directory: {config.output_dir}")
    print("=" * 60)
    
    # Setup device
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = config.device
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Initialize TensorBoard writer
    writer = None
    if config.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(config.output_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logs will be saved to {log_dir}")
        except ImportError:
            print("TensorBoard not available, disabling logging")
    
    # Build teacher model (for pseudo-label generation)
    print("\nBuilding teacher model...")
    teacher_model = RoboDepthZooWrapper(
        model_name=config.teacher_model,
        pretrained=config.use_pretrained,
        img_size=config.image_size,
        device=device,
    )
    teacher_model.eval()
    print(f"Teacher model info: {teacher_model.get_model_info()['name']}")
    print(f"Teacher parameters: {teacher_model.get_model_info()['num_parameters']:,}")
    
    # Build student model (for training)
    print("\nBuilding student model...")
    student_model = RoboDepthZooWrapper(
        model_name=config.student_model,
        pretrained=config.use_pretrained,
        img_size=config.image_size,
        device=device,
    )
    print(f"Student model info: {student_model.get_model_info()['name']}")
    print(f"Student parameters: {student_model.get_model_info()['num_parameters']:,}")
    
    # Create dataset
    print("\nCreating dataset...")
    image_dir = os.path.join(config.data_root, "images")
    
    if config.pseudo_label_mode == "precomputed":
        pseudo_label_dir = config.precomputed_labels_dir
    else:
        pseudo_label_dir = None
    
    dataset = RoboDepthPseudoLabelDataset(
        image_dir=image_dir,
        pseudo_label_dir=pseudo_label_dir,
        teacher_model=teacher_model if config.pseudo_label_mode == "online" else None,
        img_size=config.image_size,
        confidence_threshold=config.confidence_threshold,
        mode=config.pseudo_label_mode,
    )
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    
    # Initialize loss function
    criterion = RoboDepthLoss(loss_type=config.loss_type)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Initialize scheduler
    scheduler = get_learning_rate_scheduler(
        optimizer,
        max_epochs=config.max_epochs,
        warmup_epochs=config.warmup_epochs,
    )
    
    # Mixed precision setup
    use_amp = config.mixed_precision and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = float("inf")
    
    if config.resume:
        print(f"\nResuming from checkpoint: {config.resume}")
        start_epoch, _ = load_checkpoint(
            student_model,
            config.resume,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
    
    # Training loop
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(start_epoch, config.max_epochs):
        print(f"\nEpoch {epoch+1}/{config.max_epochs}")
        
        # Train for one epoch
        train_metrics = train_one_epoch(
            model=student_model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config,
            writer=writer,
            use_amp=use_amp,
            scaler=scaler,
            device=device,
        )
        
        # Validate
        val_metrics = validate(
            model=student_model,
            dataloader=val_loader,
            criterion=criterion,
            epoch=epoch,
            config=config,
            writer=writer,
            device=device,
        )
        
        # Check if this is the best model
        is_best = val_metrics["abs_rel"] < best_metric
        if is_best:
            best_metric = val_metrics["abs_rel"]
            print(f"New best model! AbsRel: {best_metric:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0 or is_best:
            save_checkpoint(
                model=student_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics},
                output_dir=config.output_dir,
                filename=f"checkpoint_epoch_{epoch+1}.pth.tar",
                is_best=is_best,
            )
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation AbsRel: {best_metric:.4f}")
    print(f"Checkpoints saved to: {config.output_dir}")
    print("=" * 60)


# Import load_checkpoint for resume functionality
def load_checkpoint(*args, **kwargs):
    from .train_utils import load_checkpoint as _load_checkpoint
    return _load_checkpoint(*args, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    config = create_config_from_args(args)
    train_robodepth_zoo(config)
