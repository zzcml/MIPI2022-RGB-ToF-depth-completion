"""
Lotus-2 Teacher Model for Pseudo-Label Generation

This module provides the Lotus2Teacher class that wraps the Lotus-2 pipeline
for generating pseudo labels (depth or normal maps) from RGB images.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
from PIL import Image
from torch import nn

logger = logging.getLogger(__name__)


class LocalContinuityModule(nn.Module):
    """Local Continuity Module (LCM) used in Lotus-2 pipeline."""
    
    def __init__(self, num_channels: int):
        super().__init__()
        self.lcm = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lcm_dtype = next(self.lcm.parameters()).dtype
        if x.dtype != lcm_dtype:
            x = x.to(dtype=lcm_dtype)
        return x + self.lcm(x)


class Lotus2Teacher:
    """
    Teacher model wrapper for Lotus-2 to generate pseudo labels.
    
    This class loads the Lotus-2 model with its adapters (core predictor,
    local continuity module, and detail sharpener) and provides methods
    to generate pseudo labels for depth or normal estimation tasks.
    """
    
    def __init__(
        self,
        pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev",
        core_predictor_model_path: Optional[str] = None,
        lcm_model_path: Optional[str] = None,
        detail_sharpener_model_path: Optional[str] = None,
        task_name: str = "depth",
        mixed_precision: str = "bf16",
        device: Optional[str] = None,
    ):
        """
        Initialize the Lotus-2 teacher model.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained Flux model or HuggingFace model ID
            core_predictor_model_path: Path to core predictor LoRA weights
            lcm_model_path: Path to local continuity module weights
            detail_sharpener_model_path: Path to detail sharpener LoRA weights
            task_name: Task type, either "depth" or "normal"
            mixed_precision: Mixed precision type ("no", "fp16", "bf16")
            device: Device to run the model on (auto-detected if None)
        """
        self.task_name = task_name
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Set dtype
        if mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32
        
        logger.info(f"Initializing Lotus-2 teacher model on {self.device} with {self.weight_dtype}")
        
        # Import required modules
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
            from diffusers.utils import convert_unet_state_dict_to_peft
            from peft import LoraConfig, set_peft_model_state_dict
            
            # Try to import from local pipeline or use default
            try:
                from .pipeline_wrapper import Lotus2PipelineWrapper
            except ImportError:
                try:
                    from pipeline import Lotus2Pipeline as Lotus2PipelineWrapper
                except ImportError:
                    Lotus2PipelineWrapper = None
                    logger.warning("Lotus2Pipeline not found. Using basic inference mode.")
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}. "
                "Please ensure diffusers and peft are available."
            )
        
        # Default model filenames
        self.core_predictor_filename = {
            "depth": "lotus-2_core_predictor_depth.safetensors",
            "normal": "lotus-2_core_predictor_normal.safetensors"
        }
        self.lcm_filename = {
            "depth": "lotus-2_lcm_depth.safetensors",
            "normal": "lotus-2_lcm_normal.safetensors"
        }
        self.detail_sharpener_filename = {
            "depth": "lotus-2_detail_sharpener_depth.safetensors",
            "normal": "lotus-2_detail_sharpener_normal.safetensors"
        }
        
        # Load scheduler and transformer
        self._load_models(
            core_predictor_model_path,
            lcm_model_path,
            detail_sharpener_model_path,
        )
        
        logger.info("Lotus-2 teacher model initialized successfully")
    
    def _load_models(
        self,
        core_predictor_model_path: Optional[str],
        lcm_model_path: Optional[str],
        detail_sharpener_model_path: Optional[str],
    ):
        """Load all required models and adapters."""
        from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
        from diffusers.utils import convert_unet_state_dict_to_peft
        from peft import LoraConfig, set_peft_model_state_dict
        
        # Determine which pipeline to use
        try:
            from .pipeline_wrapper import Lotus2PipelineWrapper
            use_custom_pipeline = True
        except ImportError:
            try:
                from pipeline import Lotus2Pipeline as Lotus2PipelineWrapper
                use_custom_pipeline = True
            except ImportError:
                use_custom_pipeline = False
        
        # Load scheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="scheduler", 
            num_train_timesteps=10
        )
        
        # Load transformer
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="transformer"
        )
        self.transformer.requires_grad_(False)
        self.transformer.to(device=self.device, dtype=self.weight_dtype)
        
        # Load LoRA rank based on task
        lora_rank = 128 if self.task_name == 'depth' else 256
        
        # Target modules for LoRA
        target_lora_modules = [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2",
        ]
        
        # Load core predictor adapter
        self._load_adapter(
            self.transformer,
            core_predictor_model_path,
            self.core_predictor_filename[self.task_name],
            "core_predictor",
            lora_rank,
            target_lora_modules,
        )
        
        # Load LCM
        self.local_continuity_module = self._load_lcm(lcm_model_path)
        
        # Load detail sharpener adapter
        self._load_adapter(
            self.transformer,
            detail_sharpener_model_path,
            self.detail_sharpener_filename[self.task_name],
            "detail_sharpener",
            lora_rank,
            target_lora_modules,
        )
        
        # Create pipeline if available
        if use_custom_pipeline:
            self.pipeline = Lotus2PipelineWrapper.from_pretrained(
                self.pretrained_model_name_or_path,
                scheduler=self.noise_scheduler,
                transformer=self.transformer,
                torch_dtype=self.weight_dtype,
            )
            self.pipeline.local_continuity_module = self.local_continuity_module
            self.pipeline = self.pipeline.to(self.device)
            self.use_pipeline = True
        else:
            self.pipeline = None
            self.use_pipeline = False
            logger.info("Using basic inference mode without custom pipeline")
    
    def _load_adapter(
        self,
        transformer: nn.Module,
        model_path: Optional[str],
        filename: str,
        adapter_name: str,
        lora_rank: int,
        target_modules: List[str],
    ):
        """Load a LoRA adapter into the transformer."""
        from diffusers.utils import convert_unet_state_dict_to_peft
        from peft import LoraConfig, set_peft_model_state_dict
        
        # Auto-download logic could be added here
        if model_path is None:
            logger.warning(f"{adapter_name} model path not provided. Attempting to use default location.")
            # In practice, you would download from HuggingFace here
            return
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        transformer.add_adapter(lora_config, adapter_name=adapter_name)
        
        # Load state dict
        try:
            from diffusers import Lotus2Pipeline
            lora_state_dict = Lotus2Pipeline.lora_state_dict(model_path)
        except Exception:
            # Fallback: load directly
            lora_state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v 
            for k, v in lora_state_dict.items() 
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        
        incompatible_keys = set_peft_model_state_dict(
            transformer, 
            transformer_state_dict, 
            adapter_name=adapter_name
        )
        
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights led to unexpected keys: {unexpected_keys}"
                )
        
        # Freeze adapter parameters
        for name, param in transformer.named_parameters():
            if adapter_name in name:
                param.requires_grad = False
        
        logger.info(f"Successfully loaded {adapter_name} adapter from {model_path}")
    
    def _load_lcm(self, lcm_model_path: Optional[str]) -> LocalContinuityModule:
        """Load the Local Continuity Module."""
        lcm = LocalContinuityModule(self.transformer.config.in_channels // 4)
        
        if lcm_model_path is not None:
            lcm_state_dict = torch.load(lcm_model_path, map_location="cpu", weights_only=True)
            lcm.load_state_dict(lcm_state_dict)
            logger.info(f"Successfully loaded LCM from {lcm_model_path}")
        else:
            logger.warning("LCM model path not provided. Using uninitialized LCM.")
        
        lcm.requires_grad_(False)
        lcm.to(device=self.device, dtype=self.weight_dtype)
        
        return lcm
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray, List[Image.Image]],
        num_inference_steps: int = 10,
        guidance_scale: float = 3.5,
        process_res: Optional[int] = None,
        output_type: str = "numpy",
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray]]:
        """
        Generate pseudo labels using the Lotus-2 teacher model.
        
        Args:
            image: Input RGB image(s). Can be PIL Image, torch tensor, numpy array, or list
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            process_res: Processing resolution (auto-calculated if None)
            output_type: Output format ("numpy", "tensor", "pil")
        
        Returns:
            Generated pseudo labels (depth or normal maps)
        """
        # Handle batched input
        if isinstance(image, list):
            results = []
            for img in image:
                result = self._predict_single(
                    img, num_inference_steps, guidance_scale, process_res, output_type
                )
                results.append(result)
            return results
        
        return self._predict_single(
            image, num_inference_steps, guidance_scale, process_res, output_type
        )
    
    def _predict_single(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        num_inference_steps: int,
        guidance_scale: float,
        process_res: Optional[int],
        output_type: str,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Predict pseudo label for a single image."""
        # Convert input to tensor
        if isinstance(image, Image.Image):
            image_np = np.array(image).astype(np.float32)
            image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor / 127.5 - 1.0
        elif isinstance(image, np.ndarray):
            image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor / 127.5 - 1.0
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image_tensor = image.unsqueeze(0)
            else:
                image_tensor = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        image_tensor = image_tensor.to(device=self.device, dtype=self.weight_dtype)
        
        # Auto-calculate process_res
        height, width = image_tensor.shape[2:]
        max_edge = max(height, width)
        if process_res is None:
            if max_edge > 1024:
                process_res = 1024
            elif max_edge < 512:
                process_res = 512
        
        if self.use_pipeline and self.pipeline is not None:
            # Use custom pipeline
            self.pipeline.set_progress_bar_config(disable=True)
            
            prediction = self.pipeline(
                rgb_in=image_tensor,
                prompt='',
                num_inference_steps=num_inference_steps,
                output_type='np' if output_type == 'numpy' else output_type,
                process_res=process_res,
                guidance_scale=guidance_scale,
            ).images[0]
        else:
            # Basic inference fallback
            prediction = self._basic_inference(
                image_tensor, num_inference_steps, guidance_scale, process_res
            )
        
        # Post-process based on task
        if self.task_name == "depth":
            if prediction.ndim == 3:
                output = prediction.mean(axis=-1)
            else:
                output = prediction
        elif self.task_name == "normal":
            output = prediction
        else:
            raise ValueError(f"Invalid task name: {self.task_name}")
        
        if output_type == "tensor":
            return torch.from_numpy(output)
        elif output_type == "pil":
            if self.task_name == "depth":
                # Normalize and convert to uint8
                output_norm = ((output - output.min()) / (output.max() - output.min()) * 255).astype(np.uint8)
                return Image.fromarray(output_norm)
            else:
                return Image.fromarray((output * 255).astype(np.uint8))
        else:  # numpy
            return output
    
    def _basic_inference(
        self,
        image_tensor: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float,
        process_res: Optional[int],
    ) -> np.ndarray:
        """Basic inference without custom pipeline (fallback)."""
        # This is a simplified version - in practice, you'd want to implement
        # the full diffusion sampling loop here
        logger.warning("Using basic inference. For best results, use the custom pipeline.")
        
        # Placeholder: return zeros
        # In a real implementation, this would run the diffusion process
        batch_size, _, height, width = image_tensor.shape
        output_height = height // 4
        output_width = width // 4
        
        if self.task_name == "depth":
            return np.zeros((output_height, output_width), dtype=np.float32)
        else:
            return np.zeros((output_height, output_width, 3), dtype=np.float32)
    
    def eval(self):
        """Set model to evaluation mode."""
        self.transformer.eval()
        self.local_continuity_module.eval()
    
    def to(self, device: Union[str, torch.device]):
        """Move model to specified device."""
        self.device = torch.device(device)
        self.transformer.to(device=self.device)
        self.local_continuity_module.to(device=self.device)
        if self.pipeline is not None:
            self.pipeline.to(device=self.device)
        return self


def load_lotus2_teacher(
    pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev",
    core_predictor_model_path: Optional[str] = None,
    lcm_model_path: Optional[str] = None,
    detail_sharpener_model_path: Optional[str] = None,
    task_name: str = "depth",
    mixed_precision: str = "bf16",
    device: Optional[str] = None,
) -> Lotus2Teacher:
    """
    Convenience function to load a Lotus-2 teacher model.
    
    Args:
        pretrained_model_name_or_path: Path to pretrained Flux model or HuggingFace model ID
        core_predictor_model_path: Path to core predictor LoRA weights
        lcm_model_path: Path to local continuity module weights
        detail_sharpener_model_path: Path to detail sharpener LoRA weights
        task_name: Task type, either "depth" or "normal"
        mixed_precision: Mixed precision type ("no", "fp16", "bf16")
        device: Device to run the model on
    
    Returns:
        Loaded Lotus2Teacher instance
    """
    return Lotus2Teacher(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        core_predictor_model_path=core_predictor_model_path,
        lcm_model_path=lcm_model_path,
        detail_sharpener_model_path=detail_sharpener_model_path,
        task_name=task_name,
        mixed_precision=mixed_precision,
        device=device,
    )
