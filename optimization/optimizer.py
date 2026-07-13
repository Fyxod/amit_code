"""
Adversarial Optimizer Module.

This module implements the main optimization loop for finding
adversarial geometric perturbations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from typing import Optional, Dict, Tuple, List, Callable
import numpy as np
from tqdm import tqdm
import os
import json

from .schedulers import create_scheduler
from .perturbation_params import PerturbationParameters
from transforms import (
    FFTPhasePerturbation,
    DelaunayWarp,
    HomographyTransform,
    PiecewiseHomography,
    ThinPlateSpline,
    RollingShutter
)


class AdversarialOptimizer(nn.Module):
    """
    Main optimizer for adversarial geometric perturbations.
    
    Combines all transformation modules and optimizes their parameters
    to generate adversarial examples while preserving perceptual quality.
    
    Args:
        config: Configuration object.
        target_model: Target model to attack.
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
        
        # Initialize transformation modules
        self._init_transforms()
        
        # Initialize loss functions
        from losses import CombinedLoss
        self.loss_fn = CombinedLoss(config, target_model, device)
        
        # Initialize optimizer and scheduler
        self._init_optimizer()
        
        # Training state
        self.current_iteration = 0
        self.best_loss = float('inf')
        self.best_params = None
        self.history = []
    
    def _init_transforms(self):
        """Initialize transformation modules."""
        h, w = self.config.image_size
        
        # FFT phase perturbation
        if self.config.fft_enabled:
            phase_resolution = getattr(self.config, 'fft_phase_resolution', None)
            shared_channels = getattr(self.config, 'fft_shared_channels', False)
            self.fft_transform = FFTPhasePerturbation(
                image_size=(h, w),
                magnitude=self.config.fft_magnitude,
                learnable=self.config.fft_learnable,
                phase_resolution=phase_resolution,
                shared_channels=shared_channels
            )
        else:
            self.fft_transform = None
        
        # Delaunay triangulation warp
        if self.config.delaunay_enabled:
            self.delaunay_transform = DelaunayWarp(
                image_size=(h, w),
                num_points=self.config.delaunay_num_points,
                max_displacement=self.config.delaunay_max_displacement,
                learnable=self.config.delaunay_learnable
            )
        else:
            self.delaunay_transform = None
        
        # Homography transformation (piecewise or global)
        if self.config.homography_enabled:
            use_piecewise = getattr(self.config, 'homography_piecewise', False)
            if use_piecewise:
                grid_size = getattr(self.config, 'homography_grid_size', (4, 4))
                self.homography_transform = PiecewiseHomography(
                    image_size=(h, w),
                    grid_size=grid_size,
                    max_perturbation=self.config.homography_max_perturbation,
                    learnable=self.config.homography_learnable
                )
            else:
                self.homography_transform = HomographyTransform(
                    image_size=(h, w),
                    max_perturbation=self.config.homography_max_perturbation,
                    learnable=self.config.homography_learnable
                )
        else:
            self.homography_transform = None
        
        # Thin-plate spline warp
        if self.config.tps_enabled:
            self.tps_transform = ThinPlateSpline(
                image_size=(h, w),
                num_control_points=self.config.tps_num_control_points,
                max_displacement=self.config.tps_max_displacement,
                learnable=self.config.tps_learnable
            )
        else:
            self.tps_transform = None
        
        # Rolling shutter effect
        if self.config.rolling_shutter_enabled:
            num_harmonics = getattr(self.config, 'rolling_shutter_num_harmonics', 2)
            per_row_amp = getattr(self.config, 'rolling_shutter_per_row_amplitude', False)
            self.rolling_shutter_transform = RollingShutter(
                image_size=(h, w),
                max_offset=self.config.rolling_shutter_max_offset,
                direction=self.config.rolling_shutter_direction,
                learnable=self.config.rolling_shutter_learnable,
                num_harmonics=num_harmonics,
                per_row_amplitude=per_row_amp
            )
        else:
            self.rolling_shutter_transform = None
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        # Collect all learnable parameters
        params = []
        for transform in [
            self.fft_transform,
            self.delaunay_transform,
            self.homography_transform,
            self.tps_transform,
            self.rolling_shutter_transform
        ]:
            if transform is not None:
                params.extend(list(transform.parameters()))
        
        # Create optimizer
        if self.config.optimizer == 'adam':
            self.optimizer = Adam(params, lr=self.config.learning_rate)
        elif self.config.optimizer == 'sgd':
            self.optimizer = SGD(params, lr=self.config.learning_rate, momentum=0.9)
        else:
            self.optimizer = Adam(params, lr=self.config.learning_rate)
        
        # Create scheduler
        if self.config.use_scheduler:
            self.scheduler = create_scheduler(
                self.optimizer,
                self.config.scheduler_type,
                step_size=self.config.step_size,
                gamma=self.config.gamma,
                T_max=self.config.num_iterations
            )
        else:
            self.scheduler = None
    
    def apply_transforms(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all enabled transformations to the input.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Transformed tensor
        """
        # Apply transformations in sequence
        if self.fft_transform is not None:
            x = self.fft_transform(x)
        
        if self.delaunay_transform is not None:
            x = self.delaunay_transform(x)
        
        if self.homography_transform is not None:
            x = self.homography_transform(x)
        
        if self.tps_transform is not None:
            x = self.tps_transform(x)
        
        if self.rolling_shutter_transform is not None:
            x = self.rolling_shutter_transform(x)
        
        return x
    
    def optimize(
        self,
        image: torch.Tensor,
        original_label: torch.Tensor = None,
        num_iterations: int = None,
        callback: Callable = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Run the optimization loop to find adversarial perturbation.
        
        Args:
            image: Input image tensor of shape (B, C, H, W)
            original_label: Original class label (optional)
            num_iterations: Number of iterations (overrides config)
            callback: Optional callback function called each iteration
            
        Returns:
            Tuple of (perturbed image, optimization results)
        """
        if num_iterations is None:
            num_iterations = self.config.num_iterations
        
        image = image.to(self.device)
        if original_label is not None:
            original_label = original_label.to(self.device)
        
        # Reset parameters
        self._reset_parameters()
        
        # Early stopping variables
        patience_counter = 0
        best_iter_loss = float('inf')
        
        # Progress bar
        pbar = tqdm(range(num_iterations), desc="Optimizing")
        
        for iteration in pbar:
            self.current_iteration = iteration
            
            # Forward pass
            perturbed = self.apply_transforms(image)
            
            # Compute loss
            total_loss, losses = self.loss_fn(
                image, perturbed, original_label, return_components=True
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Record history
            self.history.append({
                'iteration': iteration,
                'losses': {k: v.item() for k, v in losses.items()}
            })
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'adv': f'{losses.get("adversarial", torch.tensor(0)).item():.4f}'
            })
            
            # Check for improvement
            if total_loss.item() < self.best_loss:
                self.best_loss = total_loss.item()
                self.best_params = self._get_param_state()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if self.config.early_stopping and patience_counter >= self.config.patience:
                print(f"\nEarly stopping at iteration {iteration}")
                break
            
            # Callback
            if callback is not None:
                callback(iteration, losses, perturbed)
            
            # Checkpointing
            if (self.config.save_checkpoints and 
                iteration % self.config.checkpoint_interval == 0):
                self._save_checkpoint(iteration)
        
        # Load best parameters
        if self.best_params is not None:
            self._load_param_state(self.best_params)
        
        # Generate final perturbed image
        with torch.no_grad():
            final_perturbed = self.apply_transforms(image)
        
        results = {
            'best_loss': self.best_loss,
            'final_loss': total_loss.item(),
            'num_iterations': iteration + 1,
            'history': self.history
        }
        
        return final_perturbed, results
    
    def _reset_parameters(self):
        """Reset all transformation parameters."""
        for transform in [
            self.fft_transform,
            self.delaunay_transform,
            self.homography_transform,
            self.tps_transform,
            self.rolling_shutter_transform
        ]:
            if transform is not None and hasattr(transform, 'reset_parameters'):
                transform.reset_parameters()
    
    def _get_param_state(self) -> Dict:
        """Get current parameter state."""
        state = {}
        for name, param in self.named_parameters():
            state[name] = param.data.clone()
        return state
    
    def _load_param_state(self, state: Dict):
        """Load parameter state."""
        for name, param in self.named_parameters():
            if name in state:
                param.data = state[name]
    
    def _save_checkpoint(self, iteration: int):
        """Save optimization checkpoint."""
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history
        }
        
        path = os.path.join(checkpoint_dir, f'checkpoint_{iteration}.pt')
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load optimization checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_iteration = checkpoint['iteration']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
    
    def get_transform_magnitudes(self) -> Dict[str, float]:
        """Get current magnitude of each transformation."""
        magnitudes = {}
        
        if self.fft_transform is not None:
            magnitudes['fft'] = self.fft_transform.get_perturbation_magnitude()
        
        if self.delaunay_transform is not None:
            magnitudes['delaunay'] = self.delaunay_transform.get_displacement_magnitude()
        
        if self.homography_transform is not None:
            magnitudes['homography'] = self.homography_transform.get_perturbation_magnitude()
        
        if self.tps_transform is not None:
            magnitudes['tps'] = self.tps_transform.get_displacement_magnitude()
        
        if self.rolling_shutter_transform is not None:
            magnitudes['rolling_shutter'] = self.rolling_shutter_transform.get_offset_magnitude()
        
        return magnitudes


class MultiImageOptimizer(AdversarialOptimizer):
    """
    Optimizer for multiple images.
    
    Extends AdversarialOptimizer to handle batches of images
    with individual perturbation parameters.
    
    Args:
        config: Configuration object.
        target_model: Target model to attack.
        device: Device to run on.
    """
    
    def __init__(self, config, target_model: nn.Module = None, device: str = 'cuda'):
        super().__init__(config, target_model, device)
    
    def optimize_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor = None,
        num_iterations: int = None
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Optimize perturbations for a batch of images.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
            num_iterations: Number of iterations
            
        Returns:
            Tuple of (perturbed images, list of results)
        """
        batch_size = images.shape[0]
        all_perturbed = []
        all_results = []
        
        for i in range(batch_size):
            image = images[i:i+1]
            label = labels[i:i+1] if labels is not None else None
            
            perturbed, results = self.optimize(image, label, num_iterations)
            
            all_perturbed.append(perturbed)
            all_results.append(results)
        
        perturbed_batch = torch.cat(all_perturbed, dim=0)
        
        return perturbed_batch, all_results


class EnsembleOptimizer(nn.Module):
    """
    Ensemble optimizer combining multiple optimization strategies.
    
    Runs multiple optimizers with different configurations and
    selects the best result.
    
    Args:
        configs: List of configuration objects.
        target_model: Target model to attack.
        device: Device to run on.
    """
    
    def __init__(
        self,
        configs: List,
        target_model: nn.Module = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.configs = configs
        self.device = device
        
        self.optimizers = nn.ModuleList([
            AdversarialOptimizer(config, target_model, device)
            for config in configs
        ])
    
    def optimize(
        self,
        image: torch.Tensor,
        original_label: torch.Tensor = None,
        num_iterations: int = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Run ensemble optimization.
        
        Args:
            image: Input image
            original_label: Original label
            num_iterations: Number of iterations per optimizer
            
        Returns:
            Best perturbed image and combined results
        """
        best_perturbed = None
        best_loss = float('inf')
        all_results = []
        
        for i, optimizer in enumerate(self.optimizers):
            perturbed, results = optimizer.optimize(
                image, original_label, num_iterations
            )
            
            all_results.append({
                'optimizer_idx': i,
                'results': results
            })
            
            if results['best_loss'] < best_loss:
                best_loss = results['best_loss']
                best_perturbed = perturbed
        
        combined_results = {
            'best_loss': best_loss,
            'best_optimizer': min(
                range(len(all_results)),
                key=lambda i: all_results[i]['results']['best_loss']
            ),
            'all_results': all_results
        }
        
        return best_perturbed, combined_results