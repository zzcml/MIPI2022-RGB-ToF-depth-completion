"""
Configuration for pseudo-label training with Lotus-2 as teacher model.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class PseudoLabelTrainingConfig:
    """Configuration for pseudo-label training using Lotus-2 as teacher."""
    
    # Teacher model settings
    teacher_model_path: str = "black-forest-labs/FLUX.1-dev"
    teacher_core_predictor_path: Optional[str] = None
    teacher_lcm_path: Optional[str] = None
    teacher_detail_sharpener_path: Optional[str] = None
    teacher_task_name: str = "depth"  # "depth" or "normal"
    teacher_mixed_precision: str = "bf16"
    
    # Inference settings for teacher
    teacher_num_inference_steps: int = 10
    teacher_guidance_scale: float = 3.5
    teacher_process_res: Optional[int] = None
    
    # Dataset settings
    train_data_dir: str = ""
    val_data_dir: Optional[str] = None
    image_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    
    # Training settings
    output_dir: str = "./outputs"
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Loss settings
    loss_type: str = "l1"  # "l1", "l2", "smooth_l1"
    use_confidence_weighting: bool = True
    confidence_threshold: float = 0.5
    
    # Logging and checkpointing
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 1000
    
    # Device settings
    device: str = "cuda"
    num_workers: int = 4
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.teacher_task_name not in ["depth", "normal"]:
            raise ValueError(f"task_name must be 'depth' or 'normal', got {self.teacher_task_name}")
        
        if self.teacher_mixed_precision not in ["no", "fp16", "bf16"]:
            raise ValueError(f"mixed_precision must be 'no', 'fp16', or 'bf16', got {self.teacher_mixed_precision}")
        
        if self.loss_type not in ["l1", "l2", "smooth_l1"]:
            raise ValueError(f"loss_type must be 'l1', 'l2', or 'smooth_l1', got {self.loss_type}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PseudoLabelTrainingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
