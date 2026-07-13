"""
Embedding Loss Module using TAESD.

This module implements embedding-based loss using the Tiny AutoEncoder for Stable Diffusion (TAESD).
The loss measures the distance between latent embeddings of original and perturbed images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import os


class TAESDEmbeddingLoss(nn.Module):
    """
    Embedding loss using TAESD (Tiny AutoEncoder for Stable Diffusion).
    
    Uses the TAESD encoder to extract latent representations and computes
    the distance between embeddings of original and perturbed images.
    
    Args:
        taesd_path: Path to the TAESD model directory.
        device: Device to run on.
        loss_type: Type of distance metric ('l1', 'l2', 'cosine').
        weight: Weight for the loss.
    """
    
    def __init__(
        self,
        taesd_path: str = 'taesd',
        device: str = 'cuda',
        loss_type: str = 'l2',
        weight: float = 1.0
    ):
        super().__init__()
        self.device = device
        self.loss_type = loss_type
        self.weight = weight
        
        # Load TAESD encoder
        self.encoder = self._load_taesd_encoder(taesd_path)
        
        if self.encoder is not None:
            self.encoder = self.encoder.to(device)
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def _load_taesd_encoder(self, taesd_path: str) -> nn.Module:
        """Load TAESD encoder from the model files."""
        try:
            # Try loading with diffusers
            from diffusers import AutoencoderTiny
            
            encoder_path = os.path.join(taesd_path, 'taesd_encoder.safetensors')
            if os.path.exists(encoder_path):
                # Load the full autoencoder and extract encoder
                vae = AutoencoderTiny.from_pretrained(taesd_path)
                return vae.encoder
            else:
                print(f"TAESD encoder not found at {encoder_path}, using fallback")
                return self._create_fallback_encoder()
                
        except ImportError:
            print("Diffusers not available, using fallback encoder")
            return self._create_fallback_encoder()
        except Exception as e:
            print(f"Could not load TAESD: {e}, using fallback encoder")
            return self._create_fallback_encoder()
    
    def _create_fallback_encoder(self) -> nn.Module:
        """Create a simple CNN encoder as fallback."""
        return nn.Sequential(
            # Initial convolution
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Downsample blocks
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 4, 3, stride=1, padding=1),
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent representation.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Latent embedding tensor
        """
        # Normalize input to [-1, 1] if needed
        if x.min() >= 0:
            x = 2.0 * x - 1.0
        
        with torch.no_grad():
            latent = self.encoder(x)
        
        return latent
    
    def forward(
        self,
        original: torch.Tensor,
        perturbed: torch.Tensor,
        return_embedding: bool = False
    ) -> torch.Tensor:
        """
        Compute embedding loss between original and perturbed images.
        
        Args:
            original: Original image tensor of shape (B, C, H, W)
            perturbed: Perturbed image tensor of shape (B, C, H, W)
            return_embedding: If True, also return embeddings
            
        Returns:
            Loss tensor (and optionally embeddings)
        """
        # Encode both images
        orig_embedding = self.encode(original)
        pert_embedding = self.encode(perturbed)
        
        # Compute distance based on loss type
        if self.loss_type == 'l1':
            loss = F.l1_loss(orig_embedding, pert_embedding)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(orig_embedding, pert_embedding)
        elif self.loss_type == 'cosine':
            # Flatten embeddings for cosine similarity
            orig_flat = orig_embedding.view(orig_embedding.shape[0], -1)
            pert_flat = pert_embedding.view(pert_embedding.shape[0], -1)
            
            # Cosine similarity (1 - similarity = distance)
            similarity = F.cosine_similarity(orig_flat, pert_flat, dim=1)
            loss = (1 - similarity).mean()
        else:
            loss = F.mse_loss(orig_embedding, pert_embedding)
        
        weighted_loss = self.weight * loss
        
        if return_embedding:
            return weighted_loss, (orig_embedding, pert_embedding)
        return weighted_loss
    
    def get_embedding_distance(
        self,
        original: torch.Tensor,
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Get raw embedding distance between images.
        
        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            
        Returns:
            Distance tensor
        """
        orig_embedding = self.encode(original)
        pert_embedding = self.encode(perturbed)
        
        return F.mse_loss(orig_embedding, pert_embedding, reduction='none').mean(dim=(1, 2, 3))


class MultiScaleEmbeddingLoss(nn.Module):
    """
    Multi-scale embedding loss using TAESD.
    
    Computes embedding loss at multiple scales for more robust
    feature matching.
    
    Args:
        taesd_path: Path to TAESD model.
        device: Device to run on.
        scales: List of scale factors.
        weights: Weights for each scale.
    """
    
    def __init__(
        self,
        taesd_path: str = 'taesd',
        device: str = 'cuda',
        scales: list = [1.0, 0.5, 0.25],
        weights: list = None
    ):
        super().__init__()
        self.scales = scales
        
        if weights is None:
            self.weights = [1.0 / len(scales)] * len(scales)
        else:
            self.weights = weights
        
        # Base embedding loss
        self.embedding_loss = TAESDEmbeddingLoss(taesd_path=taesd_path, device=device)
    
    def forward(
        self,
        original: torch.Tensor,
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-scale embedding loss.
        
        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            
        Returns:
            Multi-scale embedding loss
        """
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale != 1.0:
                # Resize images
                h, w = original.shape[2], original.shape[3]
                new_h, new_w = int(h * scale), int(w * scale)
                
                orig_scaled = F.interpolate(original, size=(new_h, new_w), mode='bilinear', align_corners=False)
                pert_scaled = F.interpolate(perturbed, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                orig_scaled = original
                pert_scaled = perturbed
            
            loss = self.embedding_loss(orig_scaled, pert_scaled)
            total_loss = total_loss + weight * loss
        
        return total_loss


class LatentConsistencyLoss(nn.Module):
    """
    Latent consistency loss for maintaining semantic structure.
    
    Encourages the perturbed image to have similar latent structure
    to the original image while allowing for local variations.
    
    Args:
        taesd_path: Path to TAESD model.
        device: Device to run on.
        consistency_threshold: Threshold for consistency penalty.
    """
    
    def __init__(
        self,
        taesd_path: str = 'taesd',
        device: str = 'cuda',
        consistency_threshold: float = 0.1
    ):
        super().__init__()
        self.consistency_threshold = consistency_threshold
        
        self.embedding_loss = TAESDEmbeddingLoss(taesd_path=taesd_path, device=device)
    
    def forward(
        self,
        original: torch.Tensor,
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute latent consistency loss.
        
        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            
        Returns:
            Consistency loss
        """
        # Get embeddings
        orig_embedding = self.embedding_loss.encode(original)
        pert_embedding = self.embedding_loss.encode(perturbed)
        
        # Compute per-channel statistics
        orig_mean = orig_embedding.mean(dim=(2, 3))
        pert_mean = pert_embedding.mean(dim=(2, 3))
        
        orig_std = orig_embedding.std(dim=(2, 3))
        pert_std = pert_embedding.std(dim=(2, 3))
        
        # Statistics matching loss
        mean_loss = F.mse_loss(orig_mean, pert_mean)
        std_loss = F.mse_loss(orig_std, pert_std)
        
        # Spatial consistency loss (with threshold)
        spatial_diff = torch.abs(orig_embedding - pert_embedding)
        spatial_loss = F.relu(spatial_diff - self.consistency_threshold).mean()
        
        return mean_loss + std_loss + spatial_loss