"""
Configuration for RoboDepth Zoo Model Training

Defines configuration dataclass and model zoo registry for RoboDepth models.
Reference: https://github.com/worldbench/RoboDepth
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
import os


# RoboDepth Model Zoo Registry
# Based on models available in RoboDepth repository
MODEL_ZOO_REGISTRY: Dict[str, Dict] = {
    # Base models from RoboDepth
    "robodepth_resnet50": {
        "name": "ResNet-50",
        "architecture": "resnet",
        "depth": 50,
        "pretrained": True,
        "description": "ResNet-50 based depth estimation model"
    },
    "robodepth_resnet101": {
        "name": "ResNet-101",
        "architecture": "resnet",
        "depth": 101,
        "pretrained": True,
        "description": "ResNet-101 based depth estimation model"
    },
    "robodepth_dinov2_base": {
        "name": "DINOv2 Base",
        "architecture": "dinov2",
        "variant": "base",
        "pretrained": True,
        "description": "DINOv2 ViT-Base based depth estimation model"
    },
    "robodepth_dinov2_large": {
        "name": "DINOv2 Large",
        "architecture": "dinov2",
        "variant": "large",
        "pretrained": True,
        "description": "DINOv2 ViT-Large based depth estimation model (Recommended for Teacher)"
    },
    "robodepth_vit_small": {
        "name": "ViT Small",
        "architecture": "vit",
        "variant": "small",
        "pretrained": True,
        "description": "Vision Transformer Small for depth estimation"
    },
    "robodepth_vit_base": {
        "name": "ViT Base",
        "architecture": "vit",
        "variant": "base",
        "pretrained": True,
        "description": "Vision Transformer Base for depth estimation"
    },
}


@dataclass
class RoboDepthZooConfig:
    """
    Configuration class for RoboDepth Zoo model training.
    
    Attributes:
        # Model Selection
        teacher_model: Name of the teacher model from MODEL_ZOO_REGISTRY
        student_model: Name of the student model from MODEL_ZOO_REGISTRY
        use_pretrained: Whether to use pretrained weights
        
        # Data Configuration
        data_root: Root directory for dataset
        image_size: Input image size (height, width) or single int
        normalize_depth: Whether to normalize depth values
        
        # Training Configuration
        batch_size: Training batch size
        num_workers: Number of data loading workers
        max_epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        warmup_epochs: Number of warmup epochs
        
        # Pseudo-label Configuration
        pseudo_label_mode: 'precomputed' or 'online'
        precomputed_labels_dir: Directory containing precomputed pseudo-labels
        confidence_threshold: Threshold for filtering low-confidence pseudo-labels
        loss_type: Type of loss ('mse', 'silog', 'berhu', 'combined')
        
        # Logging and Checkpointing
        output_dir: Output directory for checkpoints and logs
        log_interval: Interval for logging training metrics
        save_interval: Interval for saving checkpoints
        use_tensorboard: Whether to use TensorBoard logging
        
        # Device Configuration
        device: Device to use ('cuda', 'cpu', or specific cuda device)
        mixed_precision: Whether to use mixed precision training
    """
    
    # Model Selection
    teacher_model: str = "robodepth_dinov2_large"
    student_model: str = "robodepth_resnet50"
    use_pretrained: bool = True
    
    # Data Configuration
    data_root: str = "./data"
    image_size: tuple = (448, 448)
    normalize_depth: bool = True
    
    # Training Configuration
    batch_size: int = 8
    num_workers: int = 4
    max_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    
    # Pseudo-label Configuration
    pseudo_label_mode: Literal['precomputed', 'online'] = 'online'
    precomputed_labels_dir: Optional[str] = None
    confidence_threshold: float = 0.5
    loss_type: Literal['mse', 'silog', 'berhu', 'combined'] = 'combined'
    
    # Logging and Checkpointing
    output_dir: str = "./outputs/robodepth_zoo"
    log_interval: int = 10
    save_interval: int = 5
    use_tensorboard: bool = True
    
    # Device Configuration
    device: str = "cuda"
    mixed_precision: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate model names
        if self.teacher_model not in MODEL_ZOO_REGISTRY:
            raise ValueError(
                f"Teacher model '{self.teacher_model}' not found in registry. "
                f"Available models: {list(MODEL_ZOO_REGISTRY.keys())}"
            )
        if self.student_model not in MODEL_ZOO_REGISTRY:
            raise ValueError(
                f"Student model '{self.student_model}' not found in registry. "
                f"Available models: {list(MODEL_ZOO_REGISTRY.keys())}"
            )
        
        # Validate pseudo-label mode
        if self.pseudo_label_mode == 'precomputed' and not self.precomputed_labels_dir:
            raise ValueError(
                "precomputed_labels_dir must be specified when using precomputed pseudo-label mode"
            )
        
        # Process image_size
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        if self.precomputed_labels_dir:
            os.makedirs(self.precomputed_labels_dir, exist_ok=True)
    
    def get_teacher_info(self) -> Dict:
        """Get information about the teacher model."""
        return MODEL_ZOO_REGISTRY[self.teacher_model]
    
    def get_student_info(self) -> Dict:
        """Get information about the student model."""
        return MODEL_ZOO_REGISTRY[self.student_model]
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
