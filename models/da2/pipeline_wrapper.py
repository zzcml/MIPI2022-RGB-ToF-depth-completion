"""
Pipeline Wrapper for Depth Anything V2.

This module provides a wrapper to adapt Depth Anything V2 models to work
with common pipeline interfaces (similar to diffusers or other frameworks).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from PIL import Image
import numpy as np
from pathlib import Path


class DA2PipelineWrapper:
    """
    Pipeline wrapper for Depth Anything V2 models.
    
    This class provides a unified interface for running inference with
    both teacher and student models, handling preprocessing, inference,
    and postprocessing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_type: str = "teacher",  # 'teacher' or 'student'
        device: str = "cuda",
        output_resolution: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the pipeline wrapper.
        
        Args:
            model: The model to wrap (teacher or student)
            model_type: Type of model ('teacher' or 'student')
            device: Device to run inference on
            output_resolution: Optional fixed output resolution
        """
        self.model = model
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_resolution = output_resolution
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image], np.ndarray, torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Preprocess images for model input.
        
        Args:
            images: Input images (single or batch)
            target_size: Optional target size (overrides default)
            
        Returns:
            Preprocessed tensor (B, C, H, W)
        """
        target_size = target_size or self.output_resolution or (518, 518)
        
        # Handle single image
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]
        
        # Convert to tensors
        processed = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0 else img)
            elif isinstance(img, torch.Tensor):
                img = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8) if img.max() <= 1.0 else img.cpu().numpy())
            
            # Resize
            img = img.resize(target_size, Image.BILINEAR)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            processed.append(img_tensor)
        
        # Stack batch
        images_tensor = torch.stack(processed).to(self.device)
        
        # Normalize
        images_tensor = (images_tensor - self.mean) / self.std
        
        return images_tensor
    
    def postprocess(
        self,
        depth_maps: torch.Tensor,
        original_sizes: Optional[List[Tuple[int, int]]] = None,
        normalize: bool = True,
    ) -> List[np.ndarray]:
        """
        Postprocess depth predictions.
        
        Args:
            depth_maps: Raw depth predictions (B, H, W) or (B, 1, H, W)
            original_sizes: Optional list of original image sizes for resizing
            normalize: Whether to normalize each depth map to [0, 1]
            
        Returns:
            List of depth maps as numpy arrays
        """
        # Ensure 3D
        if depth_maps.dim() == 4:
            depth_maps = depth_maps.squeeze(1)
        
        # Convert to numpy
        depth_maps = depth_maps.cpu().numpy()
        
        results = []
        for i, depth in enumerate(depth_maps):
            if normalize:
                depth_min = depth.min()
                depth_max = depth.max()
                depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
            
            # Resize to original size if provided
            if original_sizes is not None:
                from PIL import Image
                orig_h, orig_w = original_sizes[i]
                depth_img = Image.fromarray(depth)
                depth_img = depth_img.resize((orig_w, orig_h), Image.BILINEAR)
                depth = np.array(depth_img)
            
            results.append(depth)
        
        return results
    
    @torch.no_grad()
    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image], str, List[str]],
        return_confidence: bool = False,
        batch_size: int = 8,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run inference on images.
        
        Args:
            images: Input images (PIL, paths, or tensors)
            return_confidence: Whether to return confidence maps
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing:
                - 'depth': List of depth maps
                - 'confidence': List of confidence maps (if requested)
        """
        # Load images if paths are provided
        if isinstance(images, str):
            images = [Image.open(images).convert('RGB')]
        elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], str):
            images = [Image.open(path).convert('RGB') for path in images]
        
        # Store original sizes
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], Image.Image):
            original_sizes = [(img.height, img.width) for img in images]
        else:
            original_sizes = None
        
        # Preprocess
        images_tensor = self.preprocess(images)
        
        # Run inference in batches
        all_depths = []
        all_confidences = [] if return_confidence else None
        
        for i in range(0, len(images_tensor), batch_size):
            batch = images_tensor[i:i+batch_size]
            
            # Forward pass
            if hasattr(self.model, 'forward'):
                output = self.model(batch)
                if isinstance(output, dict):
                    depths = output.get('depth', output.get('predictions', batch))
                else:
                    depths = output
            else:
                # For teacher model with predict method
                if hasattr(self.model, 'predict'):
                    result = self.model.predict_batch(batch, return_confidence=return_confidence)
                    depths = result['depth']
                    if return_confidence:
                        confidences = result.get('confidence', None)
                        if confidences is not None:
                            all_confidences.append(confidences)
                else:
                    depths = batch  # Fallback
            
            all_depths.append(depths)
        
        # Concatenate results
        all_depths = torch.cat(all_depths, dim=0)
        
        # Postprocess
        depth_maps = self.postprocess(all_depths, original_sizes)
        
        result = {'depth': depth_maps}
        
        if return_confidence and all_confidences is not None:
            conf_maps = self.postprocess(torch.cat(all_confidences, dim=0), original_sizes)
            result['confidence'] = conf_maps
        
        return result
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str,
        image_names: List[str],
        save_format: str = "npy",
    ) -> List[str]:
        """
        Save inference results to disk.
        
        Args:
            results: Dictionary containing depth and optionally confidence
            output_dir: Directory to save results
            image_names: Names for saving files
            save_format: Format to save ('npy', 'png')
            
        Returns:
            List of saved file paths
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for i, name in enumerate(image_names):
            depth = results['depth'][i]
            
            if save_format == "npy":
                path = os.path.join(output_dir, f"{name}_depth.npy")
                np.save(path, depth)
                saved_paths.append(path)
                
                if 'confidence' in results:
                    conf_path = os.path.join(output_dir, f"{name}_confidence.npy")
                    np.save(conf_path, results['confidence'][i])
                    saved_paths.append(conf_path)
                    
            elif save_format == "png":
                # Convert to 16-bit PNG
                depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 65535).astype(np.uint16)
                path = os.path.join(output_dir, f"{name}_depth.png")
                Image.fromarray(depth_normalized).save(path)
                saved_paths.append(path)
        
        return saved_paths


def create_pipeline(
    model: nn.Module,
    model_type: str = "student",
    device: str = "cuda",
    **kwargs,
) -> DA2PipelineWrapper:
    """
    Factory function to create a pipeline wrapper.
    
    Args:
        model: Model to wrap
        model_type: Type of model
        device: Device to use
        **kwargs: Additional arguments for DA2PipelineWrapper
        
    Returns:
        DA2PipelineWrapper instance
    """
    return DA2PipelineWrapper(
        model=model,
        model_type=model_type,
        device=device,
        **kwargs,
    )
