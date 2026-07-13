"""
Combined Loss Module.

This module combines all loss functions into a single unified loss
for adversarial geometric perturbation optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from collections import OrderedDict

from .identity_drift import IdentityDriftLoss
from .landmark_drift import LandmarkDriftLoss
from .lpips_loss import LPIPSLoss
from .adversarial_loss import AdversarialLoss
from .embedding_loss import TAESDEmbeddingLoss


class CombinedLoss(nn.Module):
    """
    Combined loss for adversarial geometric perturbations.
    
    Combines multiple loss components with configurable weights:
    - Adversarial loss (attack the target model)
    - Identity drift loss (preserve identity)
    - Landmark drift loss (preserve facial landmarks)
    - LPIPS loss (perceptual similarity)
    - Smoothness regularization
    - Total variation regularization
    
    Args:
        config: Configuration object with loss weights and parameters.
        target_model: Target model for adversarial attack.
        device: Device to run on.
    """
    
    def __init__(
        self,
        config,
        target_model: nn.Module = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.config = config
        self.device = device
        
        # Initialize loss components
        self._init_losses(target_model)
        
        # Loss history for logging
        self.loss_history = []
    
    def _init_losses(self, target_model: nn.Module):
        """Initialize individual loss components."""
        # Adversarial loss
        self.adversarial_loss = AdversarialLoss(
            target_model=target_model,
            attack_type=self.config.adversarial_target,
            target_class=self.config.target_class,
            device=self.device
        )
        
        # Identity drift loss
        if self.config.identity_weight > 0:
            self.identity_loss = IdentityDriftLoss(
                model_name=self.config.target_model,
                device=self.device,
                threshold=self.config.identity_threshold
            )
        else:
            self.identity_loss = None
        
        # Landmark drift loss
        if self.config.landmark_weight > 0:
            self.landmark_loss = LandmarkDriftLoss(
                threshold=self.config.landmark_threshold,
                device=self.device
            )
        else:
            self.landmark_loss = None
        
        # LPIPS loss
        if self.config.lpips_weight > 0:
            self.lpips_loss = LPIPSLoss(
                net=self.config.lpips_net,
                device=self.device,
                threshold=self.config.lpips_threshold
            )
        else:
            self.lpips_loss = None
        
        # Embedding loss (TAESD)
        if hasattr(self.config, 'embedding_weight') and self.config.embedding_weight > 0:
            taesd_path = getattr(self.config, 'taesd_path', 'taesd')
            self.embedding_loss = TAESDEmbeddingLoss(
                taesd_path=taesd_path,
                device=self.device,
                loss_type=getattr(self.config, 'embedding_loss_type', 'l2'),
                weight=1.0
            )
        else:
            self.embedding_loss = None
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor,
        original_label: torch.Tensor = None,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            original_label: Original class labels (optional)
            return_components: If True, return individual loss components
            
        Returns:
            Total loss and dictionary of individual losses
        """
        losses = OrderedDict()
        
        # Adversarial loss
        if self.config.adversarial_weight > 0:
            adv_loss = self.adversarial_loss(
                original, perturbed, original_label
            )
            losses['adversarial'] = adv_loss * self.config.adversarial_weight
        
        # Identity drift loss
        if self.identity_loss is not None and self.config.identity_weight > 0:
            id_loss, id_similarity = self.identity_loss(
                original, perturbed, return_similarity=True
            )
            losses['identity'] = id_loss * self.config.identity_weight
            losses['identity_similarity'] = id_similarity
        
        # Landmark drift loss
        if self.landmark_loss is not None and self.config.landmark_weight > 0:
            lm_loss = self.landmark_loss(original, perturbed)
            losses['landmark'] = lm_loss * self.config.landmark_weight
        
        # LPIPS loss
        if self.lpips_loss is not None and self.config.lpips_weight > 0:
            lpips_loss, lpips_dist = self.lpips_loss(
                original, perturbed, return_distance=True
            )
            losses['lpips'] = lpips_loss * self.config.lpips_weight
            losses['lpips_distance'] = lpips_dist
        
        # Embedding loss (TAESD)
        if self.embedding_loss is not None and hasattr(self.config, 'embedding_weight'):
            emb_loss = self.embedding_loss(original, perturbed)
            losses['embedding'] = emb_loss * self.config.embedding_weight
        
        # Smoothness regularization
        if self.config.smoothness_weight > 0:
            smooth_loss = self._smoothness_loss(perturbed)
            losses['smoothness'] = smooth_loss * self.config.smoothness_weight
        
        # Total variation regularization
        if self.config.tv_weight > 0:
            tv_loss = self._total_variation_loss(perturbed)
            losses['total_variation'] = tv_loss * self.config.tv_weight
        
        # Compute total loss
        total_loss = sum(v for k, v in losses.items() 
                        if not k.endswith('_similarity') and not k.endswith('_distance'))
        
        losses['total'] = total_loss
        
        # Store in history
        self.loss_history.append({k: v.item() for k, v in losses.items()})
        
        if return_components:
            return total_loss, losses
        return total_loss
    
    def _smoothness_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute smoothness regularization loss."""
        # Gradient in x direction
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        # Gradient in y direction
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        
        # L1 norm of gradients
        loss = torch.abs(grad_x).mean() + torch.abs(grad_y).mean()
        
        return loss
    
    def _total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation regularization loss."""
        batch_size = x.shape[0]
        
        # TV is sum of absolute differences
        tv_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        tv_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        
        return (tv_x + tv_y) / batch_size
    
    def get_loss_history(self) -> List[Dict[str, float]]:
        """Get the history of loss values."""
        return self.loss_history
    
    def clear_history(self):
        """Clear the loss history."""
        self.loss_history = []
    
    def get_last_losses(self) -> Dict[str, float]:
        """Get the most recent loss values."""
        if self.loss_history:
            return self.loss_history[-1]
        return {}


class WeightedCombinedLoss(nn.Module):
    """
    Weighted combined loss with dynamic weight adjustment.
    
    Supports adaptive weight scheduling during optimization.
    
    Args:
        losses: Dictionary of loss modules.
        weights: Dictionary of loss weights.
        weight_schedule: Schedule for weight adjustment.
    """
    
    def __init__(
        self,
        losses: Dict[str, nn.Module],
        weights: Dict[str, float],
        weight_schedule: str = 'constant'
    ):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights
        self.weight_schedule = weight_schedule
        self.step_count = 0
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute weighted combined loss."""
        losses = {}
        
        for name, loss_fn in self.losses.items():
            weight = self._get_weight(name)
            if weight > 0:
                loss = loss_fn(original, perturbed, **kwargs)
                losses[name] = loss * weight
        
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        self.step_count += 1
        
        return total_loss, losses
    
    def _get_weight(self, name: str) -> float:
        """Get current weight for a loss component."""
        base_weight = self.weights.get(name, 0.0)
        
        if self.weight_schedule == 'constant':
            return base_weight
        elif self.weight_schedule == 'linear_decay':
            # Decay weight over time
            decay = max(0.1, 1.0 - self.step_count / 1000)
            return base_weight * decay
        elif self.weight_schedule == 'cosine_decay':
            # Cosine decay
            import math
            decay = 0.5 * (1 + math.cos(math.pi * self.step_count / 1000))
            return base_weight * max(0.1, decay)
        else:
            return base_weight


class MultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss with Pareto optimization.
    
    Balances multiple objectives using Pareto front analysis.
    
    Args:
        objectives: Dictionary of objective functions.
        reference_point: Reference point for hypervolume calculation.
    """
    
    def __init__(
        self,
        objectives: Dict[str, nn.Module],
        reference_point: Dict[str, float] = None
    ):
        super().__init__()
        self.objectives = nn.ModuleDict(objectives)
        self.reference_point = reference_point or {}
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-objective loss."""
        objective_values = {}
        
        for name, obj_fn in self.objectives.items():
            value = obj_fn(original, perturbed)
            objective_values[name] = value
        
        # Simple weighted sum (can be replaced with more sophisticated methods)
        total = sum(objective_values.values())
        
        return total, objective_values


class ConstraintLoss(nn.Module):
    """
    Loss with hard constraints.
    
    Enforces constraints by projecting onto feasible region.
    
    Args:
        base_loss: Base loss function.
        constraints: Dictionary of constraint functions.
        penalty_weights: Weights for constraint violations.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        constraints: Dict[str, nn.Module],
        penalty_weights: Dict[str, float] = None
    ):
        super().__init__()
        self.base_loss = base_loss
        self.constraints = nn.ModuleDict(constraints)
        self.penalty_weights = penalty_weights or {}
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute constrained loss."""
        # Base loss
        base = self.base_loss(original, perturbed)
        
        # Constraint penalties
        penalties = {}
        total_penalty = 0.0
        
        for name, constraint_fn in self.constraints.items():
            violation = constraint_fn(original, perturbed)
            weight = self.penalty_weights.get(name, 1.0)
            penalty = F.relu(violation) * weight
            penalties[name] = penalty
            total_penalty = total_penalty + penalty
        
        total_loss = base + total_penalty
        
        result = {'base': base, 'total': total_loss}
        result.update(penalties)
        
        return total_loss, result