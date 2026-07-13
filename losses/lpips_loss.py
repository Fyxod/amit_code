"""
LPIPS Perceptual Loss Module.

This module implements the Learned Perceptual Image Patch Similarity (LPIPS)
loss, which measures perceptual similarity between images using deep features
from pretrained networks. This is more aligned with human perception than
pixel-level losses like MSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np


class LPIPSLoss(nn.Module):
    """
    LPIPS perceptual similarity loss.
    
    Uses pretrained networks (AlexNet, VGG, or SqueezeNet) to extract features
    and compute perceptual distance between images. Lower values indicate
    more similar images.
    
    Args:
        net: Network architecture ('alex', 'vgg', 'squeeze').
        spatial: If True, return spatial map of distances.
        lpips: If True, use learned weights for feature comparison.
        pretrained: If True, load pretrained weights.
        device: Device to run on.
        threshold: Maximum allowed perceptual distance.
    """
    
    def __init__(
        self,
        net: str = 'alex',
        spatial: bool = False,
        lpips: bool = True,
        pretrained: bool = True,
        device: str = 'cuda',
        threshold: float = 0.3
    ):
        super().__init__()
        self.net_type = net
        self.spatial = spatial
        self.use_lpips = lpips
        self.device = device
        self.threshold = threshold
        
        # Try to use official LPIPS library
        self.lpips_model = None
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net=net, spatial=spatial).to(device)
            for param in self.lpips_model.parameters():
                param.requires_grad = False
        except ImportError:
            print("LPIPS library not available, using fallback implementation")
            self.lpips_model = None
        
        # Fallback feature extractor
        if self.lpips_model is None:
            self.feature_extractor = self._build_feature_extractor(net)
            self.weights = self._get_default_weights()
    
    def _build_feature_extractor(self, net: str) -> nn.Module:
        """Build feature extractor network."""
        if net == 'vgg':
            try:
                import torchvision.models as models
                vgg = models.vgg16(pretrained=True).features
                for param in vgg.parameters():
                    param.requires_grad = False
                return vgg.to(self.device)
            except Exception:
                pass
        
        # Fallback: simple CNN
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ).to(self.device)
    
    def _get_default_weights(self) -> List[float]:
        """Get default weights for feature layers."""
        return [1.0, 1.0, 1.0, 1.0, 1.0]
    
    def _normalize_image(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize image for feature extraction."""
        # Assume input is in [0, 1]
        if x.min() < 0:
            # Already normalized
            return x
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        return (x - mean) / std
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor,
        return_distance: bool = False
    ) -> torch.Tensor:
        """
        Compute LPIPS perceptual loss.
        
        Args:
            original: Original image tensor of shape (B, C, H, W)
            perturbed: Perturbed image tensor of shape (B, C, H, W)
            return_distance: If True, also return raw distance
            
        Returns:
            Loss tensor (and optionally distance)
        """
        # Normalize images
        original_norm = self._normalize_image(original)
        perturbed_norm = self._normalize_image(perturbed)
        
        if self.lpips_model is not None:
            # Use official LPIPS
            with torch.no_grad():
                distance = self.lpips_model(original_norm, perturbed_norm)
            
            if not self.spatial:
                distance = distance.mean()
        else:
            # Use fallback implementation
            distance = self._compute_lpips_fallback(original_norm, perturbed_norm)
        
        # Apply threshold
        thresholded_distance = F.relu(distance - self.threshold)
        
        if return_distance:
            return thresholded_distance, distance
        return thresholded_distance
    
    def _compute_lpips_fallback(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LPIPS using fallback feature extractor.
        
        Args:
            original: Normalized original image
            perturbed: Normalized perturbed image
            
        Returns:
            Perceptual distance
        """
        # Extract features at multiple scales
        orig_features = original
        pert_features = perturbed
        
        total_distance = 0.0
        layer_idx = 0
        
        for layer in self.feature_extractor:
            orig_features = layer(orig_features)
            pert_features = layer(pert_features)
            
            # Compute distance at pooling layers
            if isinstance(layer, nn.MaxPool2d):
                # Normalize features
                orig_norm = F.normalize(orig_features, p=2, dim=1)
                pert_norm = F.normalize(pert_features, p=2, dim=1)
                
                # L2 distance
                layer_distance = ((orig_norm - pert_norm) ** 2).mean(dim=(1, 2, 3))
                
                if layer_idx < len(self.weights):
                    total_distance = total_distance + self.weights[layer_idx] * layer_distance.mean()
                    layer_idx += 1
        
        return total_distance
    
    def get_perceptual_distance(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Get raw perceptual distance between images.
        
        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            
        Returns:
            Perceptual distance tensor
        """
        original_norm = self._normalize_image(original)
        perturbed_norm = self._normalize_image(perturbed)
        
        if self.lpips_model is not None:
            with torch.no_grad():
                distance = self.lpips_model(original_norm, perturbed_norm)
            return distance.mean()
        else:
            return self._compute_lpips_fallback(original_norm, perturbed_norm)


class PerceptualLoss(nn.Module):
    """
    General perceptual loss using deep features.
    
    Computes loss based on feature differences at multiple layers
    of a pretrained network.
    
    Args:
        model_name: Name of the pretrained model ('vgg', 'resnet').
        layers: List of layer names to extract features from.
        weights: Weights for each layer's contribution.
        device: Device to run on.
    """
    
    def __init__(
        self,
        model_name: str = 'vgg',
        layers: List[str] = None,
        weights: List[float] = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        if layers is None:
            layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']
        self.layers = layers
        
        if weights is None:
            weights = [1.0 / len(layers)] * len(layers)
        self.weights = weights
        
        # Load model
        self.model, self.layer_indices = self._load_model()
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _load_model(self) -> Tuple[nn.Module, Dict]:
        """Load pretrained model and get layer indices."""
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features.to(self.device)
            
            # Map layer names to indices
            layer_indices = {
                'conv1_1': 0,
                'conv1_2': 2,
                'conv2_1': 5,
                'conv2_2': 7,
                'conv3_1': 10,
                'conv3_2': 12,
                'conv3_3': 14,
                'conv4_1': 17,
                'conv4_2': 19,
                'conv4_3': 21,
                'conv5_1': 24,
                'conv5_2': 26,
                'conv5_3': 28,
            }
            
            return vgg, layer_indices
        except Exception:
            return self._build_simple_model()
    
    def _build_simple_model(self) -> Tuple[nn.Module, Dict]:
        """Build simple model as fallback."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ).to(self.device)
        
        layer_indices = {
            'conv1_1': 0,
            'conv1_2': 2,
            'conv2_1': 5,
            'conv2_2': 7,
        }
        
        return model, layer_indices
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            
        Returns:
            Perceptual loss
        """
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(original.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(original.device)
        
        original_norm = (original - mean) / std
        perturbed_norm = (perturbed - mean) / std
        
        # Extract features and compute loss
        total_loss = 0.0
        
        for layer_name, weight in zip(self.layers, self.weights):
            if layer_name in self.layer_indices:
                idx = self.layer_indices[layer_name]
                
                # Get features up to this layer
                orig_feat = original_norm
                pert_feat = perturbed_norm
                
                for i, layer in enumerate(self.model):
                    orig_feat = layer(orig_feat)
                    pert_feat = layer(pert_feat)
                    if i == idx:
                        break
                
                # Compute loss
                layer_loss = F.mse_loss(orig_feat, pert_feat)
                total_loss = total_loss + weight * layer_loss
        
        return total_loss


class StyleLoss(nn.Module):
    """
    Style loss based on Gram matrices.
    
    Computes the difference in style (texture/statistics) between images
    using Gram matrices of deep features.
    
    Args:
        model_name: Name of the feature extraction model.
        layers: Layers to compute style from.
        device: Device to run on.
    """
    
    def __init__(
        self,
        model_name: str = 'vgg',
        layers: List[str] = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        if layers is None:
            layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']
        self.layers = layers
        
        # Load model
        self.model, self.layer_indices = self._load_model()
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _load_model(self) -> Tuple[nn.Module, Dict]:
        """Load pretrained model."""
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features.to(self.device)
            
            layer_indices = {
                'conv1_1': 0, 'conv1_2': 2,
                'conv2_1': 5, 'conv2_2': 7,
                'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14,
                'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21,
                'conv5_1': 24, 'conv5_2': 26, 'conv5_3': 28,
            }
            
            return vgg, layer_indices
        except Exception:
            return self._build_simple_model()
    
    def _build_simple_model(self) -> Tuple[nn.Module, Dict]:
        """Build simple model as fallback."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ).to(self.device)
        
        layer_indices = {'conv1_1': 0, 'conv1_2': 2}
        return model, layer_indices
    
    def _gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix of features."""
        b, c, h, w = x.shape
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute style loss.
        
        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            
        Returns:
            Style loss
        """
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(original.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(original.device)
        
        original_norm = (original - mean) / std
        perturbed_norm = (perturbed - mean) / std
        
        total_loss = 0.0
        
        for layer_name in self.layers:
            if layer_name in self.layer_indices:
                idx = self.layer_indices[layer_name]
                
                # Get features
                orig_feat = original_norm
                pert_feat = perturbed_norm
                
                for i, layer in enumerate(self.model):
                    orig_feat = layer(orig_feat)
                    pert_feat = layer(pert_feat)
                    if i == idx:
                        break
                
                # Compute Gram matrices
                orig_gram = self._gram_matrix(orig_feat)
                pert_gram = self._gram_matrix(pert_feat)
                
                # Style loss
                layer_loss = F.mse_loss(orig_gram, pert_gram)
                total_loss = total_loss + layer_loss
        
        return total_loss / len(self.layers)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    
    Computes SSIM between images and returns 1 - SSIM as loss.
    SSIM considers luminance, contrast, and structure.
    
    Args:
        window_size: Size of the Gaussian window.
        channel: Number of input channels.
        size_average: If True, average over spatial dimensions.
    """
    
    def __init__(
        self,
        window_size: int = 11,
        channel: int = 3,
        size_average: bool = True
    ):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, channel))
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM."""
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
        
        return window
    
    def _ssim(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor
    ) -> torch.Tensor:
        """Compute SSIM between two images."""
        channel = img1.shape[1]
        
        if img1.device != self.window.device:
            self.window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(
            img1 * img1, self.window, padding=self.window_size // 2, groups=channel
        ) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, self.window, padding=self.window_size // 2, groups=channel
        ) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, self.window, padding=self.window_size // 2, groups=channel
        ) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        return ssim_map.mean(dim=(1, 2, 3))
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM loss.
        
        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            
        Returns:
            SSIM loss (1 - SSIM)
        """
        ssim_value = self._ssim(original, perturbed)
        return 1 - ssim_value