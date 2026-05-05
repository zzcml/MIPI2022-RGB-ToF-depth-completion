"""
Configuration for RoboDepth Zoo Model Training

Defines configuration dataclass and model zoo registry for RoboDepth models.
Reference: https://github.com/worldbench/RoboDepth

The RoboDepth zoo contains multiple depth estimation model implementations:
- CADepth: Context-Aware Depth estimation
- DIFFNet: Differential Image Feature Fusion Network
- DNet: Deep Neural Network for depth estimation
- DepthHints: Self-supervised depth estimation with hints
- DynaDepth: Dynamic depth estimation with IMU
- EPCDepth: Edge-Preserving Completion for depth
- FSRE-Depth: Few-Shot Representation Enhancement
- GCNDepth: Graph Convolutional Network for depth
- HR-Depth: High-Resolution depth estimation
- Insta-DM: Instant Depth Mapping
- Lite-Mono: Lightweight monocular depth estimation
- ManyDepth: Multi-frame depth estimation
- MaskOcc: Masked Occlusion handling
- MonoDepth2: Classic self-supervised depth estimation
- MonoViT: Vision Transformer for monocular depth
- PackNet-SfM: Packing networks for Structure from Motion
- RA-Depth: Resolution-Aware depth estimation
- SGDepth: Semantic-Guided depth estimation
- TriDepth: Triple-frame depth estimation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
import os


# RoboDepth Model Zoo Registry
# Based on models available in RoboDepth repository (https://github.com/worldbench/RoboDepth/tree/main/zoo)
MODEL_ZOO_REGISTRY: Dict[str, Dict] = {
    # ============================================
    # Self-Supervised Monocular Depth Estimation
    # ============================================
    
    # MonoDepth2 - Classic baseline
    "monodepth2_resnet18": {
        "name": "MonoDepth2 (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Digging into Self-Supervised Monocular Depth Estimation (ICCV 2019)",
        "zoo_path": "zoo/MonoDepth2",
        "description": "Classic self-supervised monocular depth estimation baseline"
    },
    "monodepth2_resnet50": {
        "name": "MonoDepth2 (ResNet-50)",
        "architecture": "resnet",
        "depth": 50,
        "method": "self-supervised",
        "paper": "Digging into Self-Supervised Monocular Depth Estimation (ICCV 2019)",
        "zoo_path": "zoo/MonoDepth2",
        "description": "MonoDepth2 with deeper ResNet-50 encoder"
    },
    
    # DepthHints - Self-supervised with depth hints
    "depthhints_resnet18": {
        "name": "DepthHints (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Self-Supervised Monocular Depth Hints (ICCV 2019)",
        "zoo_path": "zoo/DepthHints",
        "description": "Self-supervised depth estimation with precomputed depth hints"
    },
    
    # DNet - Deep Neural Network
    "dnet_resnet18": {
        "name": "DNet (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Deep Neural Network for Monocular Depth Estimation (IROS 2020)",
        "zoo_path": "zoo/DNet",
        "description": "Deep neural network based depth estimation"
    },
    
    # MaskOcc - Masked Occlusion handling
    "maskocc_resnet18": {
        "name": "MaskOcc (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Handling Occlusions with Masking (arXiv 2019)",
        "zoo_path": "zoo/MaskOcc",
        "description": "Self-supervised depth estimation with occlusion masking"
    },
    
    # CADepth - Context-Aware Depth
    "cadepth_resnet18": {
        "name": "CADepth (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Context-Aware Depth Estimation (3DV 2021)",
        "zoo_path": "zoo/CADepth",
        "description": "Context-aware depth estimation with spatial propagation"
    },
    
    # HR-Depth - High-Resolution
    "hrdepth_resnet18": {
        "name": "HR-Depth (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "High-Resolution Depth Estimation (AAAI 2021)",
        "zoo_path": "zoo/HR-Depth",
        "description": "High-resolution feature aggregation for depth estimation"
    },
    "hrdepth_mobilenetv3": {
        "name": "HR-Depth (MobileNetV3)",
        "architecture": "mobilenet",
        "variant": "v3",
        "method": "self-supervised",
        "paper": "High-Resolution Depth Estimation (AAAI 2021)",
        "zoo_path": "zoo/HR-Depth",
        "description": "Lightweight HR-Depth with MobileNetV3 encoder"
    },
    
    # DIFFNet - Differential Image Feature Fusion
    "diffnet_resnet18": {
        "name": "DIFFNet (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Differential Image Feature Fusion Network (BMVC 2021)",
        "zoo_path": "zoo/DIFFNet",
        "description": "Differential feature fusion for improved depth estimation"
    },
    "diffnet_hrnet": {
        "name": "DIFFNet (HRNet)",
        "architecture": "hrnet",
        "method": "self-supervised",
        "paper": "Differential Image Feature Fusion Network (BMVC 2021)",
        "zoo_path": "zoo/DIFFNet",
        "description": "DIFFNet with High-Resolution Network encoder"
    },
    
    # ManyDepth - Multi-frame
    "manydepth_resnet18": {
        "name": "ManyDepth (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "The Many Depths of Self-Supervised Learning (CVPR 2021)",
        "zoo_path": "zoo/ManyDepth",
        "description": "Multi-frame self-supervised depth estimation"
    },
    
    # FSRE-Depth - Few-Shot Representation Enhancement
    "fsre_depth_resnet18": {
        "name": "FSRE-Depth (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Few-Shot Representation Enhancement (ICCV 2021)",
        "zoo_path": "zoo/FSRE-Depth",
        "description": "Few-shot representation enhancement for depth estimation"
    },
    
    # Insta-DM - Instant Depth Mapping
    "instadm_resnet18": {
        "name": "Insta-DM (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Instant Depth Mapping (AAAI 2021)",
        "zoo_path": "zoo/Insta-DM",
        "description": "Fast instant depth mapping with temporal consistency"
    },
    
    # DynaDepth - Dynamic with IMU
    "dynadepth_resnet18": {
        "name": "DynaDepth (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised+imu",
        "paper": "Dynamic Depth Estimation with IMU (ECCV 2022)",
        "zoo_path": "zoo/DynaDepth",
        "description": "Dynamic depth estimation fused with IMU measurements"
    },
    
    # RA-Depth - Resolution-Aware
    "radepth_resnet18": {
        "name": "RA-Depth (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Resolution-Aware Depth Estimation (ECCV 2022)",
        "zoo_path": "zoo/RA-Depth",
        "description": "Resolution-aware multi-scale feature fusion"
    },
    "radepth_hrnet": {
        "name": "RA-Depth (HRNet)",
        "architecture": "hrnet",
        "method": "self-supervised",
        "paper": "Resolution-Aware Depth Estimation (ECCV 2022)",
        "zoo_path": "zoo/RA-Depth",
        "description": "RA-Depth with High-Resolution Network encoder"
    },
    
    # Lite-Mono - Lightweight
    "litemono_resnet18": {
        "name": "Lite-Mono (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Lite-Mono: Lightweight Monocular Depth Estimation (CVPR 2023)",
        "zoo_path": "zoo/Lite-Mono",
        "description": "Lightweight and efficient monocular depth estimation"
    },
    
    # MonoViT - Vision Transformer
    "monovit_mpvit": {
        "name": "MonoViT (MPViT)",
        "architecture": "vit",
        "variant": "mpvit",
        "method": "self-supervised",
        "paper": "Vision Transformer for Self-Supervised Depth Estimation (3DV 2022)",
        "zoo_path": "zoo/MonoViT",
        "description": "Multi-Path Vision Transformer for depth estimation"
    },
    
    # TriDepth - Triple-frame
    "tridepth_resnet18": {
        "name": "TriDepth (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Triple-Frame Depth Estimation (WACV 2023)",
        "zoo_path": "zoo/TriDepth",
        "description": "Triple-frame depth estimation with semantic guidance"
    },
    
    # EPCDepth - Edge-Preserving Completion
    "epcdepth_resnet18": {
        "name": "EPCDepth (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Edge-Preserving Completion for Depth Estimation (ICCV 2021)",
        "zoo_path": "zoo/EPCDepth",
        "description": "Edge-preserving depth completion with RSU modules"
    },
    
    # GCNDepth - Graph Convolutional Network
    "gcndepth_resnet18": {
        "name": "GCNDepth (ResNet-18)",
        "architecture": "resnet+gcn",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Graph Convolutional Network for Depth Estimation",
        "zoo_path": "zoo/GCNDepth",
        "description": "Graph convolutional refinement for depth estimation"
    },
    
    # SGDepth - Semantic-Guided
    "sgdepth_resnet18": {
        "name": "SGDepth (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised+semantic",
        "paper": "Semantic-Guided Depth Estimation (arXiv 2020)",
        "zoo_path": "zoo/SGDepth",
        "description": "Semantic-guided multi-task depth and segmentation"
    },
    
    # PackNet-SfM - Packing Networks
    "packnet_sfm_resnet18": {
        "name": "PackNet-SfM (ResNet-18)",
        "architecture": "resnet",
        "depth": 18,
        "method": "self-supervised",
        "paper": "Packing Networks for Structure from Motion (CVPR 2020)",
        "zoo_path": "zoo/PackNet-SfM",
        "description": "Efficient network packing for SfM-based depth estimation"
    },
    "packnet_sfm_packnet01": {
        "name": "PackNet-SfM (PackNet01)",
        "architecture": "packnet",
        "method": "self-supervised",
        "paper": "Packing Networks for Structure from Motion (CVPR 2020)",
        "zoo_path": "zoo/PackNet-SfM",
        "description": "Custom PackNet architecture for depth estimation"
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
