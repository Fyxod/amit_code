"""
Identity Drift Loss Module.

This module implements loss functions to measure and constrain identity
preservation during adversarial geometric perturbations. It uses face
recognition models to extract identity features and computes similarity
between original and perturbed images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class IdentityDriftLoss(nn.Module):
    """
    Identity drift loss for face recognition.
    
    Measures the change in identity features between original and perturbed
    images using a face recognition model. The loss encourages the perturbed
    image to maintain the same identity as the original.
    
    Args:
        model_name: Name of the face recognition model ('facenet', 'arcface').
        device: Device to run the model on.
        threshold: Maximum allowed identity drift (for thresholded loss).
        feature_layer: Layer to extract features from.
        normalize: Whether to L2-normalize features.
    """
    
    def __init__(
        self,
        model_name: str = 'facenet',
        device: str = 'cuda',
        threshold: float = 0.3,
        feature_layer: str = 'avgpool',
        normalize: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.feature_layer = feature_layer
        self.normalize = normalize
        
        # Load face recognition model
        self.model = self._load_model()
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _load_model(self) -> nn.Module:
        """Load the face recognition model."""
        if self.model_name == 'facenet':
            try:
                from facenet_pytorch import InceptionResnetV1
                model = InceptionResnetV1(pretrained='vggface2').eval()
            except ImportError:
                # Fallback to a simple CNN if facenet-pytorch not available
                model = self._create_simple_face_model()
        elif self.model_name == 'arcface':
            model = self._create_simple_face_model()
        else:
            model = self._create_simple_face_model()
        
        return model.to(self.device)
    
    def _create_simple_face_model(self) -> nn.Module:
        """Create a simple face recognition model as fallback."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            nn.Linear(512, 512)
        )
        return model
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract identity features from images.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, D)
        """
        # Ensure correct input size
        if x.shape[2] != 160 or x.shape[3] != 160:
            # Resize to expected input size
            x = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            features = self.model(x)
        
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor,
        return_similarity: bool = False
    ) -> torch.Tensor:
        """
        Compute identity drift loss.
        
        Args:
            original: Original image tensor of shape (B, C, H, W)
            perturbed: Perturbed image tensor of shape (B, C, H, W)
            return_similarity: If True, also return similarity score
            
        Returns:
            Loss tensor (and optionally similarity score)
        """
        # Extract features
        orig_features = self.extract_features(original)
        pert_features = self.extract_features(perturbed)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(orig_features, pert_features, dim=1)
        
        # Identity drift is 1 - similarity (0 when identical, 1 when orthogonal)
        drift = 1 - similarity
        
        # Apply threshold (penalize only if drift exceeds threshold)
        thresholded_drift = F.relu(drift - self.threshold)
        
        # Mean loss
        loss = thresholded_drift.mean()
        
        if return_similarity:
            return loss, similarity.mean()
        return loss
    
    def get_identity_similarity(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Get identity similarity score between original and perturbed images.
        
        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            
        Returns:
            Similarity score tensor
        """
        orig_features = self.extract_features(original)
        pert_features = self.extract_features(perturbed)
        
        similarity = F.cosine_similarity(orig_features, pert_features, dim=1)
        return similarity


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss using intermediate layer features.
    
    Computes loss based on feature differences at multiple layers,
    providing more fine-grained identity preservation.
    
    Args:
        model_name: Name of the feature extraction model.
        layers: List of layer names to extract features from.
        weights: Weights for each layer's contribution to the loss.
    """
    
    def __init__(
        self,
        model_name: str = 'vgg',
        layers: list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
        weights: list = None
    ):
        super().__init__()
        self.model_name = model_name
        self.layers = layers
        
        if weights is None:
            self.weights = [1.0 / len(layers)] * len(layers)
        else:
            self.weights = weights
        
        # Load feature extractor
        self.feature_extractor = self._load_feature_extractor()
    
    def _load_feature_extractor(self) -> nn.Module:
        """Load feature extraction model."""
        # Use VGG for feature extraction
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features
            # Freeze parameters
            for param in vgg.parameters():
                param.requires_grad = False
            return vgg
        except Exception:
            # Fallback to simple CNN
            return self._create_simple_feature_extractor()
    
    def _create_simple_feature_extractor(self) -> nn.Module:
        """Create simple feature extractor as fallback."""
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
        )
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            
        Returns:
            Feature matching loss
        """
        # Normalize images for VGG
        if self.model_name == 'vgg':
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(original.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(original.device)
            original_norm = (original - mean) / std
            perturbed_norm = (perturbed - mean) / std
        else:
            original_norm = original
            perturbed_norm = perturbed
        
        # Extract features at different layers
        total_loss = 0.0
        
        orig_features = original_norm
        pert_features = perturbed_norm
        
        layer_idx = 0
        for i, layer in enumerate(self.feature_extractor):
            orig_features = layer(orig_features)
            pert_features = layer(pert_features)
            
            # Compute loss at pooling layers
            if isinstance(layer, nn.MaxPool2d) and layer_idx < len(self.weights):
                layer_loss = F.mse_loss(orig_features, pert_features)
                total_loss = total_loss + self.weights[layer_idx] * layer_loss
                layer_idx += 1
        
        return total_loss


class ArcFaceLoss(nn.Module):
    """
    ArcFace-based identity loss.
    
    Uses ArcFace face recognition model for identity feature extraction.
    
    Args:
        model_path: Path to ArcFace model weights.
        device: Device to run on.
        threshold: Maximum allowed identity drift.
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = 'cuda',
        threshold: float = 0.3
    ):
        super().__init__()
        self.device = device
        self.threshold = threshold
        
        # Load ArcFace model
        self.model = self._load_arcface(model_path)
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _load_arcface(self, model_path: str) -> nn.Module:
        """Load ArcFace model."""
        # Create a ResNet-based face recognition model
        model = self._create_resnet_face_model()
        
        if model_path is not None:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Could not load ArcFace weights: {e}")
        
        return model.to(self.device)
    
    def _create_resnet_face_model(self) -> nn.Module:
        """Create ResNet-based face model."""
        import torchvision.models as models
        
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=True)
        
        # Modify final layer for face embedding
        resnet.fc = nn.Linear(resnet.fc.in_features, 512)
        
        return resnet
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """Compute ArcFace identity loss."""
        # Resize to expected input
        if original.shape[2] != 112 or original.shape[3] != 112:
            original = F.interpolate(original, size=(112, 112), mode='bilinear', align_corners=False)
            perturbed = F.interpolate(perturbed, size=(112, 112), mode='bilinear', align_corners=False)
        
        # Extract features
        with torch.no_grad():
            orig_features = self.model(original)
            pert_features = self.model(perturbed)
        
        # Normalize features
        orig_features = F.normalize(orig_features, p=2, dim=1)
        pert_features = F.normalize(pert_features, p=2, dim=1)
        
        # Compute similarity
        similarity = F.cosine_similarity(orig_features, pert_features, dim=1)
        
        # Loss
        drift = 1 - similarity
        loss = F.relu(drift - self.threshold).mean()
        
        return loss


class IdentityConsistencyLoss(nn.Module):
    """
    Identity consistency loss for multiple images of the same person.
    
    Ensures that multiple images of the same identity maintain consistent
    identity features after perturbation.
    
    Args:
        base_loss: Base identity loss to use.
        device: Device to run on.
    """
    
    def __init__(
        self,
        base_loss: str = 'facenet',
        device: str = 'cuda'
    ):
        super().__init__()
        self.identity_loss = IdentityDriftLoss(
            model_name=base_loss,
            device=device
        )
    
    def forward(
        self, 
        images: torch.Tensor, 
        perturbed_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute identity consistency loss.
        
        Args:
            images: Original images of shape (B, C, H, W)
            perturbed_images: Perturbed images of shape (B, C, H, W)
            
        Returns:
            Consistency loss
        """
        batch_size = images.shape[0]
        
        # Extract features for all images
        orig_features = self.identity_loss.extract_features(images)
        pert_features = self.identity_loss.extract_features(perturbed_images)
        
        # Compute pairwise similarities
        # Original images should have high similarity (same identity)
        orig_sim = torch.mm(orig_features, orig_features.t())
        
        # Perturbed images should maintain similar similarity structure
        pert_sim = torch.mm(pert_features, pert_features.t())
        
        # Loss: maintain similarity structure
        loss = F.mse_loss(orig_sim, pert_sim)
        
        return loss