"""
Dataset class for pseudo-label training using Lotus-2 as teacher.

This module provides dataset classes that:
1. Load RGB images from disk
2. Generate pseudo labels on-the-fly or pre-compute them using the teacher model
3. Return image-pseudo_label pairs for training student models
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class PseudoLabelDataset(Dataset):
    """
    Dataset for pseudo-label training.
    
    This dataset loads RGB images and either:
    1. Generates pseudo labels on-the-fly using a provided teacher model
    2. Loads pre-computed pseudo labels from disk
    
    Args:
        data_dir: Directory containing RGB images
        teacher_model: Optional teacher model for on-the-fly pseudo label generation
        pseudo_label_dir: Optional directory with pre-computed pseudo labels
        task_name: Task type ("depth" or "normal")
        image_extensions: List of valid image extensions
        transform: Optional transforms to apply to images
        max_samples: Maximum number of samples to use (None for all)
        preload_pseudo_labels: Whether to preload all pseudo labels into memory
    """
    
    def __init__(
        self,
        data_dir: str,
        teacher_model: Optional[Any] = None,
        pseudo_label_dir: Optional[str] = None,
        task_name: str = "depth",
        image_extensions: List[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        preload_pseudo_labels: bool = False,
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.teacher_model = teacher_model
        self.pseudo_label_dir = Path(pseudo_label_dir) if pseudo_label_dir else None
        self.task_name = task_name
        self.transform = transform
        self.target_transform = target_transform
        
        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        self.image_extensions = image_extensions
        
        # Collect image paths
        self.image_paths = self._collect_images()
        
        if max_samples is not None and max_samples < len(self.image_paths):
            self.image_paths = self.image_paths[:max_samples]
            logger.info(f"Limited dataset to {max_samples} samples")
        
        logger.info(f"Found {len(self.image_paths)} images in {data_dir}")
        
        # Validate setup
        if teacher_model is None and pseudo_label_dir is None:
            raise ValueError(
                "Either teacher_model or pseudo_label_dir must be provided"
            )
        
        # Preload pseudo labels if requested
        self.pseudo_labels = None
        if preload_pseudo_labels and self.pseudo_label_dir is not None:
            self.pseudo_labels = self._preload_pseudo_labels()
    
    def _collect_images(self) -> List[Path]:
        """Collect all valid image paths from the data directory."""
        image_paths = []
        
        for ext in self.image_extensions:
            # Search recursively
            image_paths.extend(self.data_dir.rglob(f"*{ext}"))
            image_paths.extend(self.data_dir.rglob(f"*{ext.upper()}"))
        
        # Sort for consistency
        image_paths = sorted(set(image_paths))
        
        return image_paths
    
    def _preload_pseudo_labels(self) -> Dict[str, np.ndarray]:
        """Preload all pseudo labels into memory."""
        if self.pseudo_label_dir is None:
            return {}
        
        pseudo_labels = {}
        for img_path in self.image_paths:
            rel_path = img_path.relative_to(self.data_dir)
            if self.task_name == "depth":
                label_path = self.pseudo_label_dir / f"{rel_path.stem}.npy"
            else:
                label_path = self.pseudo_label_dir / f"{rel_path.stem}.npy"
            
            if label_path.exists():
                pseudo_labels[str(rel_path)] = np.load(label_path)
            else:
                logger.warning(f"Pseudo label not found: {label_path}")
        
        logger.info(f"Preloaded {len(pseudo_labels)} pseudo labels")
        return pseudo_labels
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get an item from the dataset.
        
        Returns:
            tuple: (image_tensor, pseudo_label_tensor, metadata_dict)
        """
        img_path = self.image_paths[idx]
        rel_path = str(img_path.relative_to(self.data_dir))
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            # Default transform: convert to tensor and normalize
            image_np = np.array(image).astype(np.float32)
            image_tensor = torch.tensor(image_np).permute(2, 0, 1)
            image_tensor = image_tensor / 127.5 - 1.0  # Normalize to [-1, 1]
        
        # Get or generate pseudo label
        if self.pseudo_labels is not None and rel_path in self.pseudo_labels:
            # Use preloaded pseudo label
            pseudo_label = self.pseudo_labels[rel_path]
        elif self.pseudo_label_dir is not None:
            # Load from disk
            pseudo_label = self._load_pseudo_label(rel_path)
        elif self.teacher_model is not None:
            # Generate on-the-fly
            pseudo_label = self._generate_pseudo_label(image)
        else:
            raise ValueError(f"No pseudo label available for {img_path}")
        
        # Convert to tensor
        if isinstance(pseudo_label, np.ndarray):
            pseudo_label_tensor = torch.from_numpy(pseudo_label)
        else:
            pseudo_label_tensor = pseudo_label
        
        # Ensure correct shape
        if self.task_name == "depth":
            if pseudo_label_tensor.dim() == 3:
                pseudo_label_tensor = pseudo_label_tensor.mean(dim=-1)
            if pseudo_label_tensor.dim() == 2:
                pseudo_label_tensor = pseudo_label_tensor.unsqueeze(0)
        elif self.task_name == "normal":
            if pseudo_label_tensor.dim() == 2:
                pseudo_label_tensor = pseudo_label_tensor.unsqueeze(0)
        
        # Apply target transforms
        if self.target_transform is not None:
            pseudo_label_tensor = self.target_transform(pseudo_label_tensor)
        
        # Metadata
        metadata = {
            "image_path": str(img_path),
            "task_name": self.task_name,
        }
        
        return image_tensor, pseudo_label_tensor, metadata
    
    def _load_pseudo_label(self, rel_path: str) -> np.ndarray:
        """Load pseudo label from disk."""
        if self.pseudo_label_dir is None:
            raise ValueError("pseudo_label_dir not set")
        
        label_path = self.pseudo_label_dir / f"{Path(rel_path).stem}.npy"
        
        if not label_path.exists():
            raise FileNotFoundError(f"Pseudo label not found: {label_path}")
        
        return np.load(label_path)
    
    def _generate_pseudo_label(self, image: Image.Image) -> np.ndarray:
        """Generate pseudo label using teacher model."""
        if self.teacher_model is None:
            raise ValueError("teacher_model not set")
        
        return self.teacher_model.predict(image, output_type="numpy")
    
    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """Get information about an image without loading it."""
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        return {
            "path": str(img_path),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        }


class PseudoLabelDatasetWithConfidence(PseudoLabelDataset):
    """
    Extended dataset that also returns confidence scores for pseudo labels.
    
    Confidence can be estimated based on:
    1. Teacher model uncertainty (if available)
    2. Consistency across multiple inference runs
    3. Heuristic measures (e.g., depth gradient magnitude)
    """
    
    def __init__(
        self,
        *args,
        confidence_mode: str = "gradient",
        num_inference_runs: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.confidence_mode = confidence_mode
        self.num_inference_runs = num_inference_runs
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get an item with confidence score.
        
        Returns:
            tuple: (image_tensor, pseudo_label_tensor, confidence_tensor, metadata_dict)
        """
        img_path = self.image_paths[idx]
        rel_path = str(img_path.relative_to(self.data_dir))
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_np = np.array(image).astype(np.float32)
            image_tensor = torch.tensor(image_np).permute(2, 0, 1)
            image_tensor = image_tensor / 127.5 - 1.0
        
        # Get pseudo label and confidence
        if self.pseudo_labels is not None and rel_path in self.pseudo_labels:
            pseudo_label = self.pseudo_labels[rel_path]
            confidence = self._compute_confidence(pseudo_label)
        elif self.pseudo_label_dir is not None:
            pseudo_label = self._load_pseudo_label(rel_path)
            confidence = self._compute_confidence(pseudo_label)
        elif self.teacher_model is not None:
            pseudo_label, confidence = self._generate_pseudo_label_with_confidence(image)
        else:
            raise ValueError(f"No pseudo label available for {img_path}")
        
        # Convert to tensors
        if isinstance(pseudo_label, np.ndarray):
            pseudo_label_tensor = torch.from_numpy(pseudo_label)
        else:
            pseudo_label_tensor = pseudo_label
        
        if isinstance(confidence, np.ndarray):
            confidence_tensor = torch.from_numpy(confidence)
        else:
            confidence_tensor = torch.tensor(confidence)
        
        # Ensure correct shapes
        if self.task_name == "depth":
            if pseudo_label_tensor.dim() == 3:
                pseudo_label_tensor = pseudo_label_tensor.mean(dim=-1)
            if pseudo_label_tensor.dim() == 2:
                pseudo_label_tensor = pseudo_label_tensor.unsqueeze(0)
            if confidence_tensor.dim() == 2:
                confidence_tensor = confidence_tensor.unsqueeze(0)
        
        metadata = {
            "image_path": str(img_path),
            "task_name": self.task_name,
        }
        
        return image_tensor, pseudo_label_tensor, confidence_tensor, metadata
    
    def _compute_confidence(self, pseudo_label: np.ndarray) -> np.ndarray:
        """Compute confidence score for a pseudo label."""
        if self.confidence_mode == "gradient":
            # Lower gradient magnitude = higher confidence (smoother regions)
            if pseudo_label.ndim == 3 and pseudo_label.shape[-1] == 3:
                # Normal maps
                grad_x = np.gradient(pseudo_label[:, :, 0])
                grad_y = np.gradient(pseudo_label[:, :, 1])
            else:
                # Depth maps
                grad_x, grad_y = np.gradient(pseudo_label)
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            confidence = 1.0 / (1.0 + gradient_magnitude)
            return confidence.astype(np.float32)
        
        elif self.confidence_mode == "uniform":
            # Uniform confidence
            return np.ones_like(pseudo_label, dtype=np.float32) * 0.5
        
        else:
            # Default uniform confidence
            return np.ones_like(pseudo_label, dtype=np.float32) * 0.5
    
    def _generate_pseudo_label_with_confidence(
        self, 
        image: Image.Image
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate pseudo label with confidence using teacher model."""
        if self.num_inference_runs > 1 and self.teacher_model is not None:
            # Multiple runs for uncertainty estimation
            predictions = []
            for _ in range(self.num_inference_runs):
                pred = self.teacher_model.predict(image, output_type="numpy")
                predictions.append(pred)
            
            predictions = np.stack(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # Confidence = inverse of standard deviation
            confidence = 1.0 / (1.0 + std_pred)
            
            return mean_pred, confidence.astype(np.float32)
        else:
            # Single run
            pseudo_label = self.teacher_model.predict(image, output_type="numpy")
            confidence = self._compute_confidence(pseudo_label)
            return pseudo_label, confidence


def create_pseudo_label_dataset(
    data_dir: str,
    teacher_model: Optional[Any] = None,
    pseudo_label_dir: Optional[str] = None,
    task_name: str = "depth",
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_confidence: bool = False,
    **dataset_kwargs,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Create a DataLoader for pseudo-label training.
    
    Args:
        data_dir: Directory containing RGB images
        teacher_model: Optional teacher model for on-the-fly generation
        pseudo_label_dir: Optional directory with pre-computed pseudo labels
        task_name: Task type ("depth" or "normal")
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to pin memory in DataLoader
        use_confidence: Whether to use confidence-weighted dataset
        **dataset_kwargs: Additional arguments for PseudoLabelDataset
    
    Returns:
        DataLoader or tuple of (dataloader, dataset)
    """
    if use_confidence:
        dataset_class = PseudoLabelDatasetWithConfidence
    else:
        dataset_class = PseudoLabelDataset
    
    dataset = dataset_class(
        data_dir=data_dir,
        teacher_model=teacher_model,
        pseudo_label_dir=pseudo_label_dir,
        task_name=task_name,
        **dataset_kwargs,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    return dataloader, dataset
