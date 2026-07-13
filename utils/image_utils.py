"""
Image Utility Functions.

This module provides utilities for loading, saving, and preprocessing images.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Union
import os


def load_image(
    path: str,
    size: Tuple[int, int] = None,
    normalize: bool = True,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Load an image from file.
    
    Args:
        path: Path to the image file
        size: Optional size to resize to (H, W)
        normalize: If True, normalize to [0, 1]
        device: Device to load the tensor to
        
    Returns:
        Image tensor of shape (C, H, W) or (1, C, H, W)
    """
    # Load image
    img = Image.open(path).convert('RGB')
    
    # Resize if needed
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    
    # Convert to tensor
    img_array = np.array(img)
    tensor = torch.from_numpy(img_array).float()
    
    # Rearrange dimensions
    tensor = tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
    # Normalize
    if normalize:
        tensor = tensor / 255.0
    
    return tensor.to(device)


def save_image(
    tensor: torch.Tensor,
    path: str,
    denormalize: bool = True
):
    """
    Save a tensor as an image file.
    
    Args:
        tensor: Image tensor of shape (C, H, W) or (B, C, H, W)
        path: Path to save the image
        denormalize: If True, multiply by 255
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Move to CPU
    tensor = tensor.detach().cpu()
    
    # Denormalize
    if denormalize:
        tensor = tensor * 255.0
    
    # Clip values
    tensor = torch.clamp(tensor, 0, 255)
    
    # Convert to numpy
    img_array = tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    # Save
    img = Image.fromarray(img_array)
    img.save(path)


def preprocess_image(
    image: Union[torch.Tensor, np.ndarray, str],
    size: Tuple[int, int] = (224, 224),
    mean: List[float] = None,
    std: List[float] = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Preprocess an image for model input.
    
    Args:
        image: Input image (tensor, numpy array, or path)
        size: Target size (H, W)
        mean: Normalization mean
        std: Normalization std
        device: Device to load to
        
    Returns:
        Preprocessed image tensor of shape (1, C, H, W)
    """
    # Load if path
    if isinstance(image, str):
        image = load_image(image, size=size, normalize=True, device=device)
    elif isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
        if image.dim() == 3:
            image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        image = image / 255.0
    
    # Ensure batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Resize if needed
    if image.shape[2:] != size:
        image = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
    
    # Normalize
    if mean is not None and std is not None:
        mean_tensor = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(device)
        image = (image - mean_tensor) / std_tensor
    
    return image.to(device)


def postprocess_image(
    tensor: torch.Tensor,
    mean: List[float] = None,
    std: List[float] = None
) -> torch.Tensor:
    """
    Postprocess a tensor back to image format.
    
    Args:
        tensor: Input tensor (B, C, H, W) or (C, H, W)
        mean: Denormalization mean
        std: Denormalization std
        
    Returns:
        Image tensor in [0, 1] range
    """
    # Ensure batch dimension
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    # Denormalize
    if mean is not None and std is not None:
        mean_tensor = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
        std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
        tensor = tensor * std_tensor + mean_tensor
    
    # Clip to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor


def create_image_grid(
    images: torch.Tensor,
    nrow: int = 8,
    padding: int = 2
) -> torch.Tensor:
    """
    Create a grid of images.
    
    Args:
        images: Batch of images (B, C, H, W)
        nrow: Number of images per row
        padding: Padding between images
        
    Returns:
        Grid image tensor
    """
    return torchvision.utils.make_grid(images, nrow=nrow, padding=padding)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to numpy array.
    
    Args:
        tensor: Input tensor (C, H, W) or (B, C, H, W)
        
    Returns:
        Numpy array (H, W, C) or (B, H, W, C)
    """
    if tensor.dim() == 4:
        return tensor.permute(0, 2, 3, 1).cpu().numpy()
    return tensor.permute(1, 2, 0).cpu().numpy()


def numpy_to_tensor(array: np.ndarray, device: str = 'cuda') -> torch.Tensor:
    """
    Convert numpy array to tensor.
    
    Args:
        array: Input numpy array (H, W, C) or (B, H, W, C)
        device: Device to load to
        
    Returns:
        Tensor (C, H, W) or (B, C, H, W)
    """
    if array.ndim == 4:
        return torch.from_numpy(array).permute(0, 3, 1, 2).to(device)
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def resize_image(
    image: torch.Tensor,
    size: Tuple[int, int],
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Resize an image tensor.
    
    Args:
        image: Input tensor (B, C, H, W) or (C, H, W)
        size: Target size (H, W)
        mode: Interpolation mode
        
    Returns:
        Resized tensor
    """
    single_image = image.dim() == 3
    if single_image:
        image = image.unsqueeze(0)
    
    image = F.interpolate(image, size=size, mode=mode, align_corners=False)
    
    if single_image:
        image = image.squeeze(0)
    
    return image


def crop_image(
    image: torch.Tensor,
    box: Tuple[int, int, int, int]
) -> torch.Tensor:
    """
    Crop an image tensor.
    
    Args:
        image: Input tensor (C, H, W)
        box: Crop box (left, top, right, bottom)
        
    Returns:
        Cropped tensor
    """
    left, top, right, bottom = box
    return image[:, top:bottom, left:right]


def center_crop(
    image: torch.Tensor,
    size: Tuple[int, int]
) -> torch.Tensor:
    """
    Center crop an image tensor.
    
    Args:
        image: Input tensor (C, H, W)
        size: Target size (H, W)
        
    Returns:
        Cropped tensor
    """
    h, w = image.shape[1], image.shape[2]
    th, tw = size
    
    i = (h - th) // 2
    j = (w - tw) // 2
    
    return image[:, i:i+th, j:j+tw]


def pad_image(
    image: torch.Tensor,
    padding: int,
    mode: str = 'constant',
    value: float = 0
) -> torch.Tensor:
    """
    Pad an image tensor.
    
    Args:
        image: Input tensor (C, H, W)
        padding: Padding size
        mode: Padding mode
        value: Padding value for constant mode
        
    Returns:
        Padded tensor
    """
    return F.pad(image.unsqueeze(0), (padding,) * 4, mode=mode, value=value).squeeze(0)


def flip_image(image: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Flip an image tensor.
    
    Args:
        image: Input tensor (C, H, W)
        dim: Dimension to flip
        
    Returns:
        Flipped tensor
    """
    return torch.flip(image, dims=[dim])


def rotate_image(
    image: torch.Tensor,
    angle: float,
    size: Tuple[int, int] = None
) -> torch.Tensor:
    """
    Rotate an image tensor.
    
    Args:
        image: Input tensor (C, H, W)
        angle: Rotation angle in degrees
        size: Output size (optional)
        
    Returns:
        Rotated tensor
    """
    from torchvision.transforms import functional as TF
    
    # Add batch dimension
    image = image.unsqueeze(0)
    
    # Rotate
    rotated = TF.rotate(image, angle)
    
    # Remove batch dimension
    rotated = rotated.squeeze(0)
    
    return rotated


def adjust_brightness(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust brightness of an image."""
    return torch.clamp(image * factor, 0, 1)


def adjust_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust contrast of an image."""
    mean = image.mean()
    return torch.clamp((image - mean) * factor + mean, 0, 1)


def adjust_saturation(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust saturation of an image."""
    gray = image.mean(dim=0, keepdim=True).expand_as(image)
    return torch.clamp(image * factor + gray * (1 - factor), 0, 1)


def add_noise(
    image: torch.Tensor,
    noise_type: str = 'gaussian',
    std: float = 0.1
) -> torch.Tensor:
    """
    Add noise to an image.
    
    Args:
        image: Input tensor
        noise_type: Type of noise ('gaussian', 'uniform', 'speckle')
        std: Noise standard deviation
        
    Returns:
        Noisy tensor
    """
    if noise_type == 'gaussian':
        noise = torch.randn_like(image) * std
    elif noise_type == 'uniform':
        noise = (torch.rand_like(image) * 2 - 1) * std
    elif noise_type == 'speckle':
        noise = image * torch.randn_like(image) * std
    else:
        noise = torch.zeros_like(image)
    
    return torch.clamp(image + noise, 0, 1)