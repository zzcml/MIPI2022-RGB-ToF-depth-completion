"""
Pseudo-label Dataset for RoboDepth Zoo Models

Provides dataset classes for pseudo-label based training with RoboDepth zoo models.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Callable
from PIL import Image
import numpy as np
from pathlib import Path


class RoboDepthPseudoLabelDataset(Dataset):
    """
    Dataset for pseudo-label based depth estimation training.
    
    Supports two modes:
    1. Precomputed: Load pre-generated pseudo-labels from disk
    2. Online: Generate pseudo-labels on-the-fly using a teacher model
    
    Args:
        image_dir: Directory containing input images
        pseudo_label_dir: Directory containing precomputed pseudo-labels (for precomputed mode)
        teacher_model: Teacher model for generating pseudo-labels (for online mode)
        transform: Image transforms
        img_size: Target image size
        confidence_threshold: Threshold for filtering low-confidence labels
        file_extension: Extension of image files to load
    """
    
    def __init__(
        self,
        image_dir: str,
        pseudo_label_dir: Optional[str] = None,
        teacher_model: Optional[torch.nn.Module] = None,
        transform: Optional[Callable] = None,
        img_size: Tuple[int, int] = (448, 448),
        confidence_threshold: float = 0.5,
        file_extension: str = ".jpg",
        mode: str = "precomputed"
    ):
        super().__init__()
        
        self.image_dir = Path(image_dir)
        self.pseudo_label_dir = Path(pseudo_label_dir) if pseudo_label_dir else None
        self.teacher_model = teacher_model
        self.transform = transform
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.file_extension = file_extension
        self.mode = mode
        
        # Validate mode
        if mode == "precomputed" and not self.pseudo_label_dir:
            raise ValueError("pseudo_label_dir must be provided for precomputed mode")
        if mode == "online" and not self.teacher_model:
            raise ValueError("teacher_model must be provided for online mode")
        
        # Collect image paths
        self.image_paths = self._collect_image_paths()
        
        # Create default transform if none provided
        if self.transform is None:
            self.transform = self._default_transform()
        
        print(f"Loaded {len(self.image_paths)} images from {image_dir}")
    
    def _collect_image_paths(self) -> List[Path]:
        """Collect all image paths from the image directory."""
        image_paths = []
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        
        for ext in extensions:
            image_paths.extend(self.image_dir.glob(f"*{ext}"))
            image_paths.extend(self.image_dir.glob(f"*{ext.upper()}"))
        
        return sorted(image_paths)
    
    def _default_transform(self) -> Callable:
        """Create default image transform."""
        # Use basic transforms without torchvision dependency
        def transform(image):
            import torch
            import numpy as np
            from PIL import Image
            
            # Resize
            img = image.resize(self.img_size, Image.BILINEAR)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            
            return img_tensor
        
        return transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
                - image: Input image tensor
                - pseudo_label: Pseudo-label depth map
                - image_path: Path to the image
                - valid_mask: Mask indicating valid pixels
        """
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        if isinstance(self.transform, list):
            # Separate transform for image only
            image_tensor = self.transform[0](image) if hasattr(self.transform[0], '__call__') else image
        else:
            image_tensor = self.transform(image)
        
        # Get pseudo-label
        if self.mode == "precomputed":
            pseudo_label, valid_mask = self._load_pseudo_label(idx)
        else:  # online mode
            pseudo_label, valid_mask = self._generate_pseudo_label(image)
        
        return {
            "image": image_tensor,
            "pseudo_label": pseudo_label,
            "image_path": str(image_path),
            "valid_mask": valid_mask,
        }
    
    def _load_pseudo_label(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load precomputed pseudo-label from disk."""
        image_path = self.image_paths[idx]
        
        # Construct pseudo-label path (assuming same filename, different extension or directory)
        relative_path = image_path.relative_to(self.image_dir)
        label_path = self.pseudo_label_dir / relative_path.with_suffix(".npy")
        
        if not label_path.exists():
            # Try with original extension
            label_path = self.pseudo_label_dir / relative_path
        
        if not label_path.exists():
            # Generate pseudo-label on-the-fly as fallback
            image = Image.open(image_path).convert("RGB")
            return self._generate_pseudo_label(image)
        
        # Load pseudo-label
        if label_path.suffix == ".npy":
            pseudo_label = np.load(label_path)
        elif label_path.suffix in [".jpg", ".png"]:
            pseudo_label = np.array(Image.open(label_path))
            if pseudo_label.ndim == 3:
                pseudo_label = pseudo_label.mean(axis=2)  # Convert to grayscale
        else:
            raise ValueError(f"Unsupported pseudo-label format: {label_path.suffix}")
        
        # Convert to tensor
        pseudo_label = torch.from_numpy(pseudo_label).float()
        
        # Resize to match expected size
        if pseudo_label.shape[-2:] != self.img_size:
            pseudo_label = torch.nn.functional.interpolate(
                pseudo_label.unsqueeze(0).unsqueeze(0),
                size=self.img_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        
        # Create valid mask (filter out invalid depth values)
        valid_mask = (pseudo_label > 0) & torch.isfinite(pseudo_label)
        
        # Apply confidence threshold (optional: if confidence maps are available)
        # For now, use depth validity as confidence proxy
        
        return pseudo_label, valid_mask
    
    def _generate_pseudo_label(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate pseudo-label using teacher model."""
        if self.teacher_model is None:
            raise RuntimeError("Teacher model not available for online pseudo-label generation")
        
        self.teacher_model.eval()
        
        # Transform image for teacher model
        transform = self._default_transform()
        image_tensor = transform(image).unsqueeze(0)
        
        # Move to same device as teacher model
        device = next(self.teacher_model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Generate pseudo-label
        with torch.no_grad():
            pseudo_label = self.teacher_model(image_tensor)
        
        # Move back to CPU
        pseudo_label = pseudo_label.cpu().squeeze(0).squeeze(0)
        
        # Create valid mask
        valid_mask = (pseudo_label > 0) & torch.isfinite(pseudo_label)
        
        return pseudo_label, valid_mask
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        images = torch.stack([item["image"] for item in batch])
        pseudo_labels = torch.stack([item["pseudo_label"] for item in batch])
        valid_masks = torch.stack([item["valid_mask"] for item in batch])
        image_paths = [item["image_path"] for item in batch]
        
        return {
            "images": images,
            "pseudo_labels": pseudo_labels,
            "valid_masks": valid_masks,
            "image_paths": image_paths,
        }


def create_dataloader(
    dataset: RoboDepthPseudoLabelDataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the pseudo-label dataset.
    
    Args:
        dataset: RoboDepthPseudoLabelDataset instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn,
    )


def generate_pseudo_labels_for_dataset(
    image_dir: str,
    output_dir: str,
    teacher_model: torch.nn.Module,
    img_size: Tuple[int, int] = (448, 448),
    batch_size: int = 8,
    num_workers: int = 4
) -> None:
    """
    Generate pseudo-labels for an entire dataset using a teacher model.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save pseudo-labels
        teacher_model: Teacher model for generating pseudo-labels
        img_size: Target image size
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
    """
    from tqdm import tqdm
    import torchvision.transforms as transforms
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset without pseudo-labels
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Simple dataset for inference
    class InferenceDataset(Dataset):
        def __init__(self, image_dir, transform):
            self.image_dir = Path(image_dir)
            self.transform = transform
            self.image_paths = []
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                self.image_paths.extend(self.image_dir.glob(f"*{ext}"))
                self.image_paths.extend(self.image_dir.glob(f"*{ext.upper()}"))
            self.image_paths = sorted(self.image_paths)
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
            return {"image": image_tensor, "image_path": str(image_path)}
    
    dataset = InferenceDataset(image_dir, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    # Set teacher model to eval mode
    teacher_model.eval()
    device = next(teacher_model.parameters()).device
    
    print(f"Generating pseudo-labels for {len(dataset)} images...")
    
    for batch in tqdm(dataloader, desc="Generating pseudo-labels"):
        images = batch["image"].to(device)
        image_paths = batch["image_path"]
        
        with torch.no_grad():
            pseudo_labels = teacher_model(images)
        
        # Save pseudo-labels
        for i, path in enumerate(image_paths):
            rel_path = Path(path).relative_to(image_dir)
            output_path = Path(output_dir) / rel_path.with_suffix(".npy")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            depth_map = pseudo_labels[i].cpu().squeeze().numpy()
            np.save(output_path, depth_map)
    
    print(f"Pseudo-labels saved to {output_dir}")
