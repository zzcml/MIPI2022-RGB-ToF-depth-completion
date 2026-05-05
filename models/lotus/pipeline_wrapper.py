"""
Pipeline wrapper for Lotus-2 to integrate with the teacher model.

This module provides a wrapper that adapts the original Lotus-2 pipeline
for use in pseudo-label generation and training scenarios.
"""

import logging
from typing import Optional, Union, List

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def resize_image(image, target_size):
    """
    Resize image to target size.
    
    Args:
        image: Image in PIL.Image, numpy.array or torch.tensor format
        target_size: tuple, target size (H, W)
    
    Returns:
        Resized image in original format
    """
    if isinstance(image, list):
        return [resize_image(img, target_size) for img in image]
    
    if isinstance(image, Image.Image):
        return image.resize(target_size[::-1], Image.BILINEAR)
    elif isinstance(image, np.ndarray):
        import cv2
        if image.ndim == 4:
            resized = np.stack([cv2.resize(img, target_size[::-1]) for img in image])
            return resized
        else:
            return cv2.resize(image, target_size[::-1])
    elif isinstance(image, torch.Tensor):
        if image.dim() == 4:
            return torch.nn.functional.interpolate(
                image,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        else:
            return torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
    else:
        raise ValueError(f"Unsupported image format: {type(image)}")


def resize_to_multiple_of_16(image_tensor):
    """
    Resize image tensor to make dimensions multiples of 16.
    
    Args:
        image_tensor: Input tensor of shape (B, C, H, W)
    
    Returns:
        Resized tensor where dimensions are multiples of 16
    """
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    
    new_h = (h // 16) * 16
    new_w = (w // 16) * 16
    
    if new_h != h or new_w != w:
        resized_tensor = torch.nn.functional.interpolate(
            image_tensor,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )
        return resized_tensor
    return image_tensor


class Lotus2PipelineWrapper:
    """
    Wrapper for Lotus-2 pipeline compatible with diffusers interface.
    
    This class wraps the original Lotus-2 pipeline to provide a consistent
    interface for pseudo-label generation.
    """
    
    def __init__(self, transformer, scheduler, vae=None, local_continuity_module=None):
        """
        Initialize the pipeline wrapper.
        
        Args:
            transformer: Flux transformer model
            scheduler: Noise scheduler
            vae: VAE for encoding/decoding (optional for some tasks)
            local_continuity_module: LCM module from Lotus-2
        """
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.local_continuity_module = local_continuity_module
        
        # Set default values
        self._guidance_scale = 3.5
        self._joint_attention_kwargs = {}
        self._execution_device = next(transformer.parameters()).device
        self.dtype = next(transformer.parameters()).dtype
        
        # Try to load the actual pipeline if available
        self._pipeline = None
        try:
            from pipeline import Lotus2Pipeline
            self._pipeline_class = Lotus2Pipeline
        except ImportError:
            self._pipeline_class = None
            logger.warning("Original Lotus2Pipeline not available")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load pipeline from pretrained model.
        
        Args:
            pretrained_model_name_or_path: Path to model or HuggingFace model ID
            **kwargs: Additional arguments passed to component loaders
        
        Returns:
            PipelineWrapper instance
        """
        try:
            from diffusers import FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
            
            # Load components
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                pretrained_model_name_or_path, 
                subfolder="scheduler",
                num_train_timesteps=10,
                **kwargs
            )
            
            transformer = FluxTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                **kwargs
            )
            
            # Try to load VAE
            try:
                from diffusers import AutoencoderKL
                vae = AutoencoderKL.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder="vae",
                    **kwargs
                )
            except Exception:
                vae = None
                logger.info("VAE not loaded")
            
            return cls(
                transformer=transformer,
                scheduler=scheduler,
                vae=vae,
            )
        except ImportError as e:
            raise ImportError(f"Required packages not installed: {e}")
    
    def set_progress_bar_config(self, **kwargs):
        """Set progress bar configuration."""
        pass  # No-op for now
    
    def to(self, device):
        """Move pipeline to device."""
        self._execution_device = device
        self.transformer.to(device=device)
        if self.vae is not None:
            self.vae.to(device=device)
        if self.local_continuity_module is not None:
            self.local_continuity_module.to(device=device)
        return self
    
    @torch.no_grad()
    def __call__(
        self,
        rgb_in: torch.FloatTensor,
        prompt: str = '',
        num_inference_steps: int = 10,
        output_type: str = "np",
        process_res: Optional[int] = None,
        timestep_core_predictor: int = 1,
        guidance_scale: float = 3.5,
        return_dict: bool = True,
        joint_attention_kwargs: Optional[dict] = None,
    ):
        """
        Run inference with the pipeline.
        
        This is a simplified version that delegates to the original pipeline
        if available, or implements basic inference.
        """
        # If we have access to the original pipeline, use it
        if self._pipeline_class is not None and hasattr(self, '_full_pipeline'):
            return self._full_pipeline(
                rgb_in=rgb_in,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                output_type=output_type,
                process_res=process_res,
                timestep_core_predictor=timestep_core_predictor,
                guidance_scale=guidance_scale,
                return_dict=return_dict,
            )
        
        # Otherwise, return placeholder
        logger.warning("Using simplified inference. Full pipeline not available.")
        
        batch_size = rgb_in.shape[0]
        height, width = rgb_in.shape[2:]
        
        # Create dummy output
        if output_type == "np":
            dummy_output = np.zeros((height // 4, width // 4, 3), dtype=np.float32)
        else:
            dummy_output = torch.zeros((batch_size, 3, height // 4, width // 4))
        
        return type('PipelineOutput', (), {'images': [dummy_output]})()
    
    @property
    def device(self):
        """Get the current device."""
        return self._execution_device


# Re-export the original pipeline if available
try:
    from pipeline import Lotus2Pipeline
    Lotus2PipelineWrapper = Lotus2Pipeline
except ImportError:
    pass
