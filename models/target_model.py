"""
Target Model Wrapper Module.

This module provides wrappers for target models used in adversarial attacks.
Supports face recognition models and general classification models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np


class TargetModel(nn.Module):
    """
    Base wrapper for target models.
    
    Provides a unified interface for different model architectures
    to be used as targets for adversarial attacks.
    
    Args:
        model: The underlying model to wrap.
        model_type: Type of model ('classification', 'embedding', 'detection').
        input_size: Expected input size (H, W).
        normalize: Whether to normalize inputs.
        device: Device to run on.
    """
    
    def __init__(
        self,
        model: nn.Module = None,
        model_type: str = 'classification',
        input_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.input_size = input_size
        self.normalize = normalize
        self.device = device
        
        if self.model is not None:
            self.model = self.model.to(device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Model output
        """
        # Resize if needed
        if x.shape[2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)

        # Normalize if needed
        if self.normalize:
            x = self._normalize_input(x)
        
        return self.model(x)
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for the model."""
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std
    
    def get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        output = self.forward(x)
        if self.model_type == 'classification':
            return output.argmax(dim=1)
        return output
    
    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction confidence."""
        output = self.forward(x)
        if self.model_type == 'classification':
            return F.softmax(output, dim=1).max(dim=1)[0]
        return output
    
    def load_model(self, path: str):
        """Load model weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False


class FaceRecognitionModel(TargetModel):
    """
    Wrapper for face recognition models.
    
    Supports popular face recognition models like FaceNet, ArcFace, etc.
    
    Args:
        model_name: Name of the face recognition model.
        model_path: Path to model weights.
        input_size: Expected input size.
        device: Device to run on.
    """
    
    def __init__(
        self,
        model_name: str = 'facenet',
        model_path: str = None,
        input_size: Tuple[int, int] = (160, 160),
        device: str = 'cuda'
    ):
        self.model_name = model_name
        super().__init__(
            model=None,
            model_type='embedding',
            input_size=input_size,
            normalize=False,
            device=device
        )
        
        # Load the specified model
        self.model = self._load_model(model_name, model_path)
        
        if self.model is not None:
            self.model = self.model.to(device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
    
    def _load_model(self, model_name: str, model_path: str) -> nn.Module:
        """Load the specified face recognition model."""
        if model_name == 'facenet':
            return self._load_facenet(model_path)
        elif model_name == 'arcface':
            return self._load_arcface(model_path)
        else:
            return self._create_simple_model()
    
    def _load_facenet(self, model_path: str) -> nn.Module:
        """Load FaceNet model."""
        try:
            from facenet_pytorch import InceptionResnetV1
            model = InceptionResnetV1(pretrained='vggface2')
            self.input_size = (160, 160)
            return model
        except ImportError:
            print("facenet-pytorch not available, using fallback model")
            return self._create_simple_model()
    
    def _load_arcface(self, model_path: str) -> nn.Module:
        """Load ArcFace model."""
        try:
            import torchvision.models as models
            # Use ResNet as backbone
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 512)
            self.input_size = (112, 112)
            
            if model_path is not None:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
            
            return model
        except Exception as e:
            print(f"Could not load ArcFace: {e}")
            return self._create_simple_model()
    
    def _create_simple_model(self) -> nn.Module:
        """Create a simple face recognition model as fallback."""
        return nn.Sequential(
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract face embeddings.
        
        Args:
            x: Input face image tensor (B, C, H, W)
            
        Returns:
            Face embedding tensor (B, D)
        """
        # Resize if needed
        if x.shape[2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)

        if self.model_name == 'facenet':
            x = x * 2.0 - 1.0
        
        # The recognition weights are frozen, but the forward pass must remain
        # differentiable with respect to ``x``.  Wrapping this call in
        # ``torch.no_grad()`` silently disconnects every image/geometry
        # parameter from an embedding-space objective.
        embedding = self.model(x)
        
        # Normalize embeddings
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def get_similarity(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between two face images.
        
        Args:
            x1: First face image
            x2: Second face image
            
        Returns:
            Similarity score (cosine similarity)
        """
        emb1 = self.forward(x1)
        emb2 = self.forward(x2)
        
        return F.cosine_similarity(emb1, emb2, dim=1)
    
    def verify(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor, 
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verify if two faces belong to the same person.
        
        Args:
            x1: First face image
            x2: Second face image
            threshold: Similarity threshold for verification
            
        Returns:
            Tuple of (is_same, similarity)
        """
        similarity = self.get_similarity(x1, x2)
        is_same = (similarity > threshold).float()
        
        return is_same, similarity


class ClassificationModel(TargetModel):
    """
    Wrapper for classification models.
    
    Args:
        model_name: Name of pretrained model ('resnet50', 'vgg16', etc.).
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        device: Device to run on.
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = 1000,
        pretrained: bool = True,
        device: str = 'cuda'
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load model
        model = self._load_model(model_name, num_classes, pretrained)
        
        super().__init__(
            model=model,
            model_type='classification',
            input_size=(512, 512),
            normalize=True,
            device=device
        )
    
    def _load_model(
        self, 
        model_name: str, 
        num_classes: int, 
        pretrained: bool
    ) -> nn.Module:
        """Load pretrained classification model."""
        try:
            import torchvision.models as models
            
            if model_name == 'resnet50':
                model = models.resnet50(pretrained=pretrained)
                if num_classes != 1000:
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'resnet18':
                model = models.resnet18(pretrained=pretrained)
                if num_classes != 1000:
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'vgg16':
                model = models.vgg16(pretrained=pretrained)
                if num_classes != 1000:
                    model.classifier[6] = nn.Linear(4096, num_classes)
            else:
                model = self._create_simple_model(num_classes)
            
            return model
        except Exception as e:
            print(f"Could not load {model_name}: {e}")
            return self._create_simple_model(num_classes)
    
    def _create_simple_model(self, num_classes: int) -> nn.Module:
        """Create a simple classification model."""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def get_top_predictions(
        self, 
        x: torch.Tensor, 
        k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k predictions.
        
        Args:
            x: Input image
            k: Number of top predictions
            
        Returns:
            Tuple of (indices, probabilities)
        """
        output = self.forward(x)
        probs = F.softmax(output, dim=1)
        
        top_probs, top_indices = probs.topk(k, dim=1)
        
        return top_indices, top_probs


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models.
    
    Combines predictions from multiple models for more robust attacks.
    
    Args:
        models: List of models to ensemble.
        weights: Weights for each model's prediction.
        device: Device to run on.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: List[float] = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.device = device
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
        
        # Freeze all models
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ensemble forward pass."""
        outputs = []
        
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Weighted average
        if outputs[0].dim() == 2 and outputs[0].shape[1] > 1:
            # Classification: average logits
            ensemble_output = sum(w * o for w, o in zip(self.weights, outputs))
        else:
            # Embedding: average embeddings
            ensemble_output = sum(w * o for w, o in zip(self.weights, outputs))
            ensemble_output = F.normalize(ensemble_output, p=2, dim=1)
        
        return ensemble_output


def create_target_model(
    model_type: str,
    model_name: str = None,
    device: str = 'cuda',
    **kwargs
) -> TargetModel:
    """
    Factory function to create target models.
    
    Args:
        model_type: Type of model ('face', 'classification', 'custom').
        model_name: Name of specific model.
        device: Device to run on.
        **kwargs: Additional arguments.
        
    Returns:
        Target model wrapper
    """
    if model_type == 'face':
        return FaceRecognitionModel(
            model_name=model_name or 'facenet',
            device=device,
            **kwargs
        )
    elif model_type == 'classification':
        return ClassificationModel(
            model_name=model_name or 'resnet50',
            device=device,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
