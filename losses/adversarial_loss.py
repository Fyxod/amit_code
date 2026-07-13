"""
Adversarial Loss Module.

This module implements adversarial loss functions for attacking face recognition
and image classification models. The loss encourages the target model to produce
incorrect predictions on the perturbed images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for attacking target models.
    
    Computes loss that encourages the target model to produce incorrect predictions.
    Supports both targeted and untargeted attacks.
    
    Args:
        target_model: The model to attack.
        attack_type: Type of attack ('untargeted', 'targeted', 'dodging').
        target_class: Target class for targeted attacks.
        confidence_threshold: Confidence threshold for successful attack.
        device: Device to run on.
    """
    
    def __init__(
        self,
        target_model: nn.Module = None,
        attack_type: str = 'untargeted',
        target_class: int = None,
        confidence_threshold: float = 0.5,
        device: str = 'cuda'
    ):
        super().__init__()
        self.target_model = target_model
        self.attack_type = attack_type
        self.target_class = target_class
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Freeze target model
        if self.target_model is not None:
            for param in self.target_model.parameters():
                param.requires_grad = False
    
    def set_target_model(self, model: nn.Module):
        """Set the target model."""
        self.target_model = model
        for param in self.target_model.parameters():
            param.requires_grad = False
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor,
        original_label: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute adversarial loss.
        
        Args:
            original: Original image tensor of shape (B, C, H, W)
            perturbed: Perturbed image tensor of shape (B, C, H, W)
            original_label: Original class labels (for untargeted attacks)
            
        Returns:
            Adversarial loss
        """
        if self.target_model is None:
            return torch.tensor(0.0, device=self.device)
        
        # Get model predictions on perturbed image
        with torch.no_grad():
            original_output = self.target_model(original)
        
        perturbed_output = self.target_model(perturbed)
        
        if self.attack_type == 'untargeted':
            loss = self._untargeted_loss(
                original_output, perturbed_output, original_label
            )
        elif self.attack_type == 'targeted':
            loss = self._targeted_loss(perturbed_output)
        elif self.attack_type == 'dodging':
            loss = self._dodging_loss(original_output, perturbed_output)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
        
        return loss
    
    def _untargeted_loss(
        self, 
        original_output: torch.Tensor,
        perturbed_output: torch.Tensor,
        original_label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute untargeted adversarial loss.
        
        Maximize the loss for the original class (make model predict wrong class).
        """
        if original_label is None:
            # Use predicted class from original output
            original_label = original_output.argmax(dim=1)
        
        # Cross-entropy loss (we want to maximize this, so return negative)
        loss = F.cross_entropy(perturbed_output, original_label)
        
        # For adversarial attack, we want to maximize the loss
        # So we return negative (minimizing negative = maximizing original)
        return -loss
    
    def _targeted_loss(self, perturbed_output: torch.Tensor) -> torch.Tensor:
        """
        Compute targeted adversarial loss.
        
        Minimize the loss for the target class (make model predict target class).
        """
        if self.target_class is None:
            raise ValueError("Target class must be specified for targeted attack")
        
        target_labels = torch.full(
            (perturbed_output.shape[0],), 
            self.target_class, 
            dtype=torch.long, 
            device=perturbed_output.device
        )
        
        # Cross-entropy loss (we want to minimize this)
        loss = F.cross_entropy(perturbed_output, target_labels)
        
        return loss
    
    def _dodging_loss(
        self, 
        original_output: torch.Tensor,
        perturbed_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dodging loss for face recognition.
        
        Make the model think the perturbed image is a different person.
        """
        # For face recognition, outputs are usually embeddings
        # We want to maximize the distance between original and perturbed embeddings
        
        if original_output.dim() == 2 and original_output.shape[1] > 1:
            # Classification output - use cross-entropy
            original_label = original_output.argmax(dim=1)
            loss = F.cross_entropy(perturbed_output, original_label)
            return -loss
        else:
            # Embedding output - use cosine similarity
            similarity = F.cosine_similarity(original_output, perturbed_output, dim=1)
            
            # We want to minimize similarity (maximize distance)
            return similarity.mean()
    
    def get_attack_success_rate(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor,
        original_label: torch.Tensor = None
    ) -> float:
        """
        Get the success rate of the adversarial attack.
        
        Args:
            original: Original images
            perturbed: Perturbed images
            original_label: Original class labels
            
        Returns:
            Success rate (0 to 1)
        """
        if self.target_model is None:
            return 0.0
        
        with torch.no_grad():
            original_output = self.target_model(original)
            perturbed_output = self.target_model(perturbed)
        
        if self.attack_type == 'untargeted':
            if original_label is None:
                original_label = original_output.argmax(dim=1)
            perturbed_label = perturbed_output.argmax(dim=1)
            success = (perturbed_label != original_label).float().mean()
        
        elif self.attack_type == 'targeted':
            perturbed_label = perturbed_output.argmax(dim=1)
            target = torch.full_like(perturbed_label, self.target_class)
            success = (perturbed_label == target).float().mean()
        
        elif self.attack_type == 'dodging':
            # For face recognition
            similarity = F.cosine_similarity(original_output, perturbed_output, dim=1)
            success = (similarity < self.confidence_threshold).float().mean()
        
        else:
            success = torch.tensor(0.0)
        
        return success.item()


class FaceRecognitionAdversarialLoss(nn.Module):
    """
    Adversarial loss specifically for face recognition models.
    
    Targets face recognition models by manipulating the embedding space
    to cause misidentification.
    
    Args:
        target_model: Face recognition model.
        attack_mode: Attack mode ('dodging', 'impersonation').
        target_identity: Target identity embedding for impersonation.
        similarity_threshold: Threshold for successful dodging.
        device: Device to run on.
    """
    
    def __init__(
        self,
        target_model: nn.Module = None,
        attack_mode: str = 'dodging',
        target_identity: torch.Tensor = None,
        similarity_threshold: float = 0.5,
        device: str = 'cuda'
    ):
        super().__init__()
        self.target_model = target_model
        self.attack_mode = attack_mode
        self.target_identity = target_identity
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        if self.target_model is not None:
            for param in self.target_model.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute face recognition adversarial loss.
        
        Args:
            original: Original face image
            perturbed: Perturbed face image
            
        Returns:
            Adversarial loss
        """
        if self.target_model is None:
            return torch.tensor(0.0, device=self.device)
        
        # Get embeddings
        with torch.no_grad():
            original_embedding = self.target_model(original)
        
        perturbed_embedding = self.target_model(perturbed)
        
        # Normalize embeddings
        original_embedding = F.normalize(original_embedding, p=2, dim=1)
        perturbed_embedding = F.normalize(perturbed_embedding, p=2, dim=1)
        
        if self.attack_mode == 'dodging':
            # Maximize distance from original identity
            similarity = F.cosine_similarity(
                original_embedding, perturbed_embedding, dim=1
            )
            loss = similarity.mean()  # Minimize similarity
            
        elif self.attack_mode == 'impersonation':
            # Minimize distance to target identity
            if self.target_identity is None:
                raise ValueError("Target identity required for impersonation attack")
            
            target_embedding = self.target_identity.to(perturbed.device)
            target_embedding = F.normalize(target_embedding, p=2, dim=1)
            
            similarity = F.cosine_similarity(
                perturbed_embedding, target_embedding, dim=1
            )
            loss = -similarity.mean()  # Maximize similarity to target
        
        else:
            raise ValueError(f"Unknown attack mode: {self.attack_mode}")
        
        return loss
    
    def get_identity_similarity(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """Get identity similarity between original and perturbed images."""
        if self.target_model is None:
            return torch.tensor(0.0)
        
        with torch.no_grad():
            original_embedding = self.target_model(original)
            perturbed_embedding = self.target_model(perturbed)
        
        original_embedding = F.normalize(original_embedding, p=2, dim=1)
        perturbed_embedding = F.normalize(perturbed_embedding, p=2, dim=1)
        
        similarity = F.cosine_similarity(
            original_embedding, perturbed_embedding, dim=1
        )
        
        return similarity


class CWLoss(nn.Module):
    """
    Carlini-Wagner (CW) adversarial loss.
    
    Implements the CW attack loss which is more effective than
    standard cross-entropy for adversarial attacks.
    
    Args:
        kappa: Confidence margin for the attack.
        targeted: Whether the attack is targeted.
    """
    
    def __init__(self, kappa: float = 0.0, targeted: bool = False):
        super().__init__()
        self.kappa = kappa
        self.targeted = targeted
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CW loss.
        
        Args:
            logits: Model output logits (B, num_classes)
            labels: Target labels (for targeted) or true labels (for untargeted)
            
        Returns:
            CW loss
        """
        # Get the logit for the target/true class
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        correct_logit = (logits * one_hot).sum(dim=1)
        
        # Get the maximum logit for other classes
        other_logits = logits - 1e4 * one_hot  # Mask out correct class
        max_other_logit = other_logits.max(dim=1)[0]
        
        if self.targeted:
            # For targeted: make target class logit > others by kappa
            loss = F.relu(max_other_logit - correct_logit + self.kappa)
        else:
            # For untargeted: make correct class logit < others by kappa
            loss = F.relu(correct_logit - max_other_logit + self.kappa)
        
        return loss.mean()


class MarginLoss(nn.Module):
    """
    Margin-based adversarial loss.
    
    Encourages a margin between the correct class and other classes.
    
    Args:
        margin: Margin between correct and incorrect classes.
        targeted: Whether the attack is targeted.
    """
    
    def __init__(self, margin: float = 1.0, targeted: bool = False):
        super().__init__()
        self.margin = margin
        self.targeted = targeted
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute margin loss.
        
        Args:
            logits: Model output logits
            labels: Class labels
            
        Returns:
            Margin loss
        """
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        correct_logit = (logits * one_hot).sum(dim=1)
        
        other_logits = logits - 1e4 * one_hot
        max_other_logit = other_logits.max(dim=1)[0]
        
        if self.targeted:
            # Minimize: max_other - correct + margin
            loss = F.relu(max_other_logit - correct_logit + self.margin)
        else:
            # Maximize: correct - max_other + margin
            loss = F.relu(correct_logit - max_other_logit + self.margin)
        
        return loss.mean()


class FeatureAdversarialLoss(nn.Module):
    """
    Feature-level adversarial loss.
    
    Attacks the model at the feature level rather than output level,
    which can be more effective for some models.
    
    Args:
        target_model: Model to attack.
        feature_layer: Layer to extract features from.
        attack_type: Type of feature attack ('maximize', 'minimize', 'random').
        device: Device to run on.
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        feature_layer: str = 'avgpool',
        attack_type: str = 'maximize',
        device: str = 'cuda'
    ):
        super().__init__()
        self.target_model = target_model
        self.feature_layer = feature_layer
        self.attack_type = attack_type
        self.device = device
        
        # Hook for feature extraction
        self.features = None
        self._register_hook()
        
        for param in self.target_model.parameters():
            param.requires_grad = False
    
    def _register_hook(self):
        """Register forward hook for feature extraction."""
        def hook(module, input, output):
            self.features = output
        
        # Find the target layer
        for name, module in self.target_model.named_modules():
            if name == self.feature_layer:
                module.register_forward_hook(hook)
                break
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature adversarial loss.
        
        Args:
            original: Original image
            perturbed: Perturbed image
            
        Returns:
            Feature adversarial loss
        """
        # Get original features
        with torch.no_grad():
            _ = self.target_model(original)
            original_features = self.features.clone()
        
        # Get perturbed features
        _ = self.target_model(perturbed)
        perturbed_features = self.features
        
        if self.attack_type == 'maximize':
            # Maximize feature distance
            loss = -F.mse_loss(perturbed_features, original_features)
        elif self.attack_type == 'minimize':
            # Minimize feature distance (for impersonation)
            loss = F.mse_loss(perturbed_features, original_features)
        elif self.attack_type == 'random':
            # Move features toward random target
            random_target = torch.randn_like(original_features)
            loss = F.mse_loss(perturbed_features, random_target)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
        
        return loss