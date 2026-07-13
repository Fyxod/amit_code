"""
Perturbation Parameters Module.

This module defines learnable perturbation parameters that combine
all geometric transformation types into a unified parameter set.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
import numpy as np


class PerturbationParameters(nn.Module):
    """
    Unified perturbation parameters for all geometric transformations.
    
    Combines learnable parameters for:
    - FFT phase perturbation
    - Delaunay triangulation warp
    - Homography transformation
    - Thin-plate spline warp
    - Rolling shutter effect
    
    Args:
        config: Configuration object with transform parameters.
        image_size: Tuple of (height, width) for the images.
    """
    
    def __init__(
        self,
        config,
        image_size: Tuple[int, int] = (224, 224)
    ):
        super().__init__()
        self.config = config
        self.image_size = image_size
        
        # Initialize all perturbation parameters
        self._init_fft_params()
        self._init_delaunay_params()
        self._init_homography_params()
        self._init_tps_params()
        self._init_rolling_shutter_params()
    
    def _init_fft_params(self):
        """Initialize FFT phase perturbation parameters."""
        if self.config.fft_enabled and self.config.fft_learnable:
            h, w = self.image_size
            self.fft_phase = nn.Parameter(
                torch.zeros(3, h, w),  # 3 channels
                requires_grad=True
            )
        else:
            self.register_buffer('fft_phase', torch.zeros(3, *self.image_size))
    
    def _init_delaunay_params(self):
        """Initialize Delaunay triangulation parameters."""
        if self.config.delaunay_enabled and self.config.delaunay_learnable:
            num_points = self.config.delaunay_num_points
            self.delaunay_displacement = nn.Parameter(
                torch.zeros(num_points, 2),
                requires_grad=True
            )
        else:
            self.register_buffer('delaunay_displacement', torch.zeros(16, 2))
    
    def _init_homography_params(self):
        """Initialize homography parameters."""
        if self.config.homography_enabled and self.config.homography_learnable:
            self.homography_delta = nn.Parameter(
                torch.zeros(8),
                requires_grad=True
            )
        else:
            self.register_buffer('homography_delta', torch.zeros(8))
    
    def _init_tps_params(self):
        """Initialize thin-plate spline parameters."""
        if self.config.tps_enabled and self.config.tps_learnable:
            num_points = self.config.tps_num_control_points
            self.tps_displacement = nn.Parameter(
                torch.zeros(num_points, 2),
                requires_grad=True
            )
        else:
            self.register_buffer('tps_displacement', torch.zeros(16, 2))
    
    def _init_rolling_shutter_params(self):
        """Initialize rolling shutter parameters."""
        if self.config.rolling_shutter_enabled and self.config.rolling_shutter_learnable:
            self.rs_amplitude = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.rs_frequency = nn.Parameter(torch.ones(1), requires_grad=True)
            self.rs_phase = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.register_buffer('rs_amplitude', torch.zeros(1))
            self.register_buffer('rs_frequency', torch.ones(1))
            self.register_buffer('rs_phase', torch.zeros(1))
    
    def get_fft_phase(self) -> torch.Tensor:
        """Get bounded FFT phase perturbation."""
        return torch.tanh(self.fft_phase) * self.config.fft_magnitude
    
    def get_delaunay_displacement(self) -> torch.Tensor:
        """Get bounded Delaunay displacement."""
        return torch.tanh(self.delaunay_displacement) * self.config.delaunay_max_displacement
    
    def get_homography_matrix(self) -> torch.Tensor:
        """Get homography matrix from parameters."""
        H = torch.eye(3, device=self.homography_delta.device)
        delta = torch.tanh(self.homography_delta) * self.config.homography_max_perturbation
        
        H[0, 0] = 1 + delta[0]
        H[0, 1] = delta[1]
        H[0, 2] = delta[2]
        H[1, 0] = delta[3]
        H[1, 1] = 1 + delta[4]
        H[1, 2] = delta[5]
        H[2, 0] = delta[6]
        H[2, 1] = delta[7]
        
        return H
    
    def get_tps_displacement(self) -> torch.Tensor:
        """Get bounded TPS displacement."""
        return torch.tanh(self.tps_displacement) * self.config.tps_max_displacement
    
    def get_rolling_shutter_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get bounded rolling shutter parameters."""
        amplitude = torch.tanh(self.rs_amplitude) * self.config.rolling_shutter_max_offset
        frequency = torch.abs(self.rs_frequency) + 0.1
        phase = self.rs_phase * 2 * np.pi
        return amplitude, frequency, phase
    
    def get_all_params(self) -> Dict[str, torch.Tensor]:
        """Get all perturbation parameters as dictionary."""
        return {
            'fft_phase': self.get_fft_phase(),
            'delaunay_displacement': self.get_delaunay_displacement(),
            'homography_matrix': self.get_homography_matrix(),
            'tps_displacement': self.get_tps_displacement(),
            'rolling_shutter': self.get_rolling_shutter_params()
        }
    
    def get_param_magnitudes(self) -> Dict[str, float]:
        """Get magnitude of each perturbation parameter."""
        magnitudes = {}
        
        if self.config.fft_enabled:
            magnitudes['fft'] = self.get_fft_phase().abs().mean().item()
        
        if self.config.delaunay_enabled:
            magnitudes['delaunay'] = self.get_delaunay_displacement().norm(dim=1).mean().item()
        
        if self.config.homography_enabled:
            magnitudes['homography'] = torch.tanh(self.homography_delta).norm().item()
        
        if self.config.tps_enabled:
            magnitudes['tps'] = self.get_tps_displacement().norm(dim=1).mean().item()
        
        if self.config.rolling_shutter_enabled:
            magnitudes['rolling_shutter'] = torch.tanh(self.rs_amplitude).abs().item()
        
        return magnitudes
    
    def reset_parameters(self):
        """Reset all parameters to zero."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                nn.init.zeros_(param)
    
    def randomize(self, scale: float = 0.5):
        """Randomize all parameters."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                nn.init.normal_(param, mean=0, std=scale)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss on parameters."""
        reg_loss = 0.0
        
        # L2 regularization on all parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                reg_loss = reg_loss + (param ** 2).mean()
        
        return reg_loss


class TransformParameterGroup(nn.Module):
    """
    Group of parameters for a single transform type.
    
    Args:
        name: Name of the transform.
        param_shape: Shape of the parameter tensor.
        max_magnitude: Maximum magnitude for the parameters.
        learnable: Whether parameters are learnable.
        regularization: Regularization weight.
    """
    
    def __init__(
        self,
        name: str,
        param_shape: Tuple[int, ...],
        max_magnitude: float = 1.0,
        learnable: bool = True,
        regularization: float = 0.01
    ):
        super().__init__()
        self.name = name
        self.max_magnitude = max_magnitude
        self.regularization = regularization
        
        if learnable:
            self.params = nn.Parameter(
                torch.zeros(*param_shape),
                requires_grad=True
            )
        else:
            self.register_buffer('params', torch.zeros(*param_shape))
    
    def get_bounded_params(self) -> torch.Tensor:
        """Get parameters bounded to [-max_magnitude, max_magnitude]."""
        return torch.tanh(self.params) * self.max_magnitude
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Get regularization loss for this parameter group."""
        return self.regularization * (self.params ** 2).mean()


class HierarchicalPerturbationParams(nn.Module):
    """
    Hierarchical perturbation parameters with coarse-to-fine structure.
    
    Allows optimization at multiple scales for better convergence.
    
    Args:
        config: Configuration object.
        image_size: Image dimensions.
        num_levels: Number of hierarchy levels.
    """
    
    def __init__(
        self,
        config,
        image_size: Tuple[int, int] = (224, 224),
        num_levels: int = 3
    ):
        super().__init__()
        self.config = config
        self.image_size = image_size
        self.num_levels = num_levels
        
        # Create parameter groups at each level
        self.levels = nn.ModuleList()
        for level in range(num_levels):
            scale = 2 ** (num_levels - level - 1)  # Coarse to fine
            level_params = self._create_level_params(level, scale)
            self.levels.append(level_params)
        
        self.current_level = 0
    
    def _create_level_params(self, level: int, scale: int) -> nn.ModuleDict:
        """Create parameters for a single hierarchy level."""
        h, w = self.image_size
        h_level, w_level = h // scale, w // scale
        
        params = nn.ModuleDict()
        
        # FFT phase at this level
        if self.config.fft_enabled:
            params['fft'] = nn.Parameter(
                torch.zeros(3, h_level, w_level),
                requires_grad=True
            )
        
        # Control points at this level
        if self.config.tps_enabled:
            num_points = max(4, self.config.tps_num_control_points // scale)
            params['tps'] = nn.Parameter(
                torch.zeros(num_points, 2),
                requires_grad=True
            )
        
        return params
    
    def get_upsampled_params(self, level: int) -> Dict[str, torch.Tensor]:
        """Get parameters upsampled to full resolution."""
        import torch.nn.functional as F
        
        h, w = self.image_size
        params = {}
        
        level_params = self.levels[level]
        
        if 'fft' in level_params:
            fft = level_params['fft']
            if fft.shape[-2:] != (h, w):
                fft = F.interpolate(
                    fft.unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0)
            params['fft'] = torch.tanh(fft) * self.config.fft_magnitude
        
        if 'tps' in level_params:
            params['tps'] = torch.tanh(level_params['tps']) * self.config.tps_max_displacement
        
        return params
    
    def set_level(self, level: int):
        """Set the current optimization level."""
        self.current_level = min(level, self.num_levels - 1)
    
    def get_current_params(self) -> Dict[str, torch.Tensor]:
        """Get parameters at the current level."""
        return self.get_upsampled_params(self.current_level)
    
    def refine(self):
        """Move to the next finer level."""
        if self.current_level < self.num_levels - 1:
            self.current_level += 1
            return True
        return False