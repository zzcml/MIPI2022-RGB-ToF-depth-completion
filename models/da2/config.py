"""
Configuration for Depth Anything V2 pseudo-label training.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Tuple
import os


@dataclass
class DA2TrainingConfig:
    """Configuration for Depth Anything V2 pseudo-label training."""
    
    # Teacher model settings
    teacher_model_type: Literal["vitl"] = "vitl"  # Only ViT-L as teacher
    teacher_checkpoint_path: Optional[str] = None  # Path to teacher checkpoint
    
    # Student model settings
    student_model_type: Literal["vits", "vitb", "custom"] = "vits"
    student_checkpoint_path: Optional[str] = None  # Path to student checkpoint (for fine-tuning)
    student_pretrained: bool = True  # Use pretrained weights for student
    
    # Dataset settings
    image_dir: str = "./data/images"  # Directory containing images for pseudo-label generation
    pseudo_label_dir: str = "./data/pseudo_labels"  # Directory to save/load pseudo-labels
    precompute_pseudo_labels: bool = True  # Whether to precompute pseudo-labels or generate on-the-fly
    image_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    
    # Training settings
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    warmup_epochs: int = 5
    
    # Loss settings
    loss_type: Literal["mse", "silog", "combined"] = "combined"
    confidence_threshold: float = 0.0  # Minimum confidence for pseudo-labels
    use_confidence_weighting: bool = True  # Weight loss by teacher confidence
    
    # Optimization settings
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    scheduler: Literal["cosine", "step", "linear"] = "cosine"
    gradient_clip: float = 1.0
    
    # Logging and checkpointing
    log_dir: str = "./logs/da2_training"
    checkpoint_dir: str = "./checkpoints/da2_training"
    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 1
    
    # Hardware settings
    device: str = "cuda"  # cuda or cpu
    mixed_precision: bool = True
    amp_dtype: Literal["float16", "bfloat16"] = "float16"
    
    # Pseudo-label generation settings
    output_resolution: Tuple[int, int] = (518, 518)  # Default resolution for DA-V2
    normalize_depth: bool = True  # Normalize depth to [0, 1]
    save_as_numpy: bool = True  # Save pseudo-labels as .npy files
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate teacher model type
        if self.teacher_model_type != "vitl":
            raise ValueError(f"Teacher model must be 'vitl' for Depth Anything V2, got '{self.teacher_model_type}'")
        
        # Validate student model type
        if self.student_model_type not in ["vits", "vitb", "custom"]:
            raise ValueError(f"Student model must be 'vits', 'vitb', or 'custom', got '{self.student_model_type}'")
        
        # Create necessary directories
        os.makedirs(self.pseudo_label_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set default checkpoint paths if not provided
        if self.teacher_checkpoint_path is None:
            self.teacher_checkpoint_path = f"depth_anything_v2_{self.teacher_model_type}.pth"
        
        if self.student_checkpoint_path is None and not self.student_pretrained:
            raise ValueError("student_checkpoint_path must be provided when student_pretrained=False")


# Model architecture specifications based on Depth Anything V2
MODEL_SPECS = {
    "vits": {
        "embed_dim": 32,
        "depth": 4,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "global_pool_size": 14,
        "checkpoint_url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
    },
    "vitb": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "global_pool_size": 14,
        "checkpoint_url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"
    },
    "vitl": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "global_pool_size": 14,
        "checkpoint_url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
    }
}
