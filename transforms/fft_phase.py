"""
FFT Phase Perturbation Module.

This module implements differentiable phase perturbations in the frequency domain.
The phase of an image's FFT is perturbed while preserving the magnitude,
creating subtle but effective geometric distortions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
from typing import Optional, Tuple


class FFTPhasePerturbation(nn.Module):
    """
    Differentiable FFT phase perturbation.
    
    Applies learnable phase shifts in the frequency domain while preserving
    the magnitude spectrum. This creates geometric distortions that are
    difficult to detect but effective for adversarial attacks.
    
    Supports coarse-grid parameterization with bilinear upsampling to
    dramatically reduce the number of learnable parameters, and optional
    channel sharing.
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        magnitude: Maximum phase perturbation magnitude in radians.
        learnable: If True, phase perturbation is a learnable parameter.
        frequency_range: Tuple of (low, high) frequency cutoff (0-1).
        channels: Number of input channels (default: 3).
        phase_resolution: Tuple of (ph, pw) for coarse grid resolution.
            If None, uses full image resolution (original behavior).
            If specified, phase is stored on a small grid and upsampled.
        shared_channels: If True, use a single channel parameterized phase
            and broadcast to all channels (3x parameter reduction).
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        magnitude: float = 0.5,
        learnable: bool = True,
        frequency_range: Tuple[float, float] = (0.1, 0.9),
        channels: int = 3,
        phase_resolution: Optional[Tuple[int, int]] = None,
        shared_channels: bool = False
    ):
        super().__init__()
        self.image_size = image_size
        self.magnitude = magnitude
        self.learnable = learnable
        self.frequency_range = frequency_range
        self.channels = channels
        self.phase_resolution = phase_resolution
        self.shared_channels = shared_channels
        
        # Determine effective number of parameter channels
        self.param_channels = 1 if shared_channels else channels
        
        # Initialize phase perturbation parameters
        self._init_phase_parameters()
        
        # Create frequency mask for band-limited perturbation
        self._create_frequency_mask()
    
    def _init_phase_parameters(self):
        """Initialize learnable phase perturbation parameters."""
        if self.phase_resolution is not None:
            # Coarse grid mode: store on small grid, upsample at forward time
            ph, pw = self.phase_resolution
            param_shape = (self.param_channels, ph, pw)
        else:
            # Full resolution mode (original behavior)
            h, w = self.image_size
            param_shape = (self.param_channels, h, w)
        
        if self.learnable:
            self.phase_param = nn.Parameter(
                torch.zeros(*param_shape),
                requires_grad=True
            )
        else:
            self.register_buffer(
                'phase_param',
                torch.zeros(*param_shape)
            )
    
    def _create_frequency_mask(self):
        """Create a frequency mask to limit perturbation to specific frequency bands."""
        h, w = self.image_size
        
        # Create frequency grid
        y_freq = torch.fft.fftfreq(h).abs()
        x_freq = torch.fft.fftfreq(w).abs()
        
        # Create 2D frequency grid
        y_grid, x_grid = torch.meshgrid(y_freq, x_freq, indexing='ij')
        
        # Normalized frequency magnitude (0 to 1)
        freq_magnitude = torch.sqrt(x_grid ** 2 + y_grid ** 2)
        freq_magnitude = freq_magnitude / freq_magnitude.max()
        
        # Create band-pass mask
        low, high = self.frequency_range
        mask = ((freq_magnitude >= low) & (freq_magnitude <= high)).float()
        
        # Smooth the mask edges
        mask = self._smooth_mask(mask)
        
        self.register_buffer('frequency_mask', mask)
    
    def _smooth_mask(self, mask: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
        """Apply Gaussian smoothing to mask edges for smoother transitions."""
        from scipy.ndimage import gaussian_filter
        import numpy as np
        
        mask_np = mask.numpy()
        smoothed = gaussian_filter(mask_np, sigma=sigma * min(self.image_size))
        
        return torch.from_numpy(smoothed).float()
    
    def get_phase_perturbation(self) -> torch.Tensor:
        """
        Get the current phase perturbation values.
        
        Returns:
            Phase perturbation tensor of shape (C, H, W)
        """
        # Apply tanh to bound the phase perturbation
        phase = torch.tanh(self.phase_param) * self.magnitude
        
        # Upsample from coarse grid if needed
        if self.phase_resolution is not None:
            h, w = self.image_size
            # phase shape: (param_channels, ph, pw) -> (param_channels, H, W)
            phase = F.interpolate(
                phase.unsqueeze(0),  # (1, param_channels, ph, pw)
                size=(h, w),
                mode='bilinear',
                align_corners=True
            ).squeeze(0)  # (param_channels, H, W)
        
        # Expand shared channels to all channels if needed
        if self.shared_channels:
            # phase shape: (1, H, W) -> (C, H, W)
            phase = phase.expand(self.channels, -1, -1)
        
        # Apply frequency mask
        phase = phase * self.frequency_mask.unsqueeze(0)
        
        return phase
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply FFT phase perturbation to input images.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Perturbed tensor of shape (B, C, H, W)
        """
        batch_size = x.shape[0]
        
        # Get phase perturbation
        phase_shift = self.get_phase_perturbation()
        
        # Apply FFT to each image in the batch
        output = []
        for i in range(batch_size):
            perturbed = self._apply_phase_perturbation(x[i], phase_shift)
            output.append(perturbed)
        
        return torch.stack(output)
    
    def _apply_phase_perturbation(
        self, 
        img: torch.Tensor, 
        phase_shift: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply phase perturbation to a single image.
        
        Args:
            img: Input image tensor of shape (C, H, W)
            phase_shift: Phase shift tensor of shape (C, H, W)
            
        Returns:
            Perturbed image tensor of shape (C, H, W)
        """
        # Compute 2D FFT
        fft = torch.fft.fft2(img, dim=(-2, -1))
        
        # Get magnitude and phase
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)
        
        # Apply phase perturbation
        perturbed_phase = phase + phase_shift.to(phase.device)
        
        # Reconstruct complex FFT
        perturbed_fft = magnitude * torch.exp(1j * perturbed_phase)
        
        # Inverse FFT
        perturbed_img = torch.fft.ifft2(perturbed_fft, dim=(-2, -1)).real
        
        return perturbed_img
    
    def reset_parameters(self):
        """Reset phase parameters to zero."""
        if self.learnable:
            nn.init.zeros_(self.phase_param)
    
    def randomize(self, magnitude: Optional[float] = None):
        """
        Randomize phase perturbation parameters.
        
        Args:
            magnitude: Optional magnitude override for randomization
        """
        mag = magnitude if magnitude is not None else self.magnitude
        
        if self.learnable:
            with torch.no_grad():
                self.phase_param.data = torch.randn_like(self.phase_param) * mag
        else:
            self.phase_param = torch.randn_like(self.phase_param) * mag
    
    def get_perturbation_magnitude(self) -> float:
        """Get the current magnitude of phase perturbation."""
        return torch.abs(self.get_phase_perturbation()).mean().item()
    
    def extra_repr(self) -> str:
        return (f'image_size={self.image_size}, magnitude={self.magnitude}, '
                f'learnable={self.learnable}, frequency_range={self.frequency_range}')


class MultiScaleFFTPhasePerturbation(nn.Module):
    """
    Multi-scale FFT phase perturbation.
    
    Applies phase perturbations at multiple frequency scales for more
    diverse and effective geometric distortions.
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        num_scales: Number of frequency scales to use.
        base_magnitude: Base magnitude for phase perturbation.
        learnable: If True, phase perturbations are learnable.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        num_scales: int = 3,
        base_magnitude: float = 0.5,
        learnable: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.num_scales = num_scales
        self.base_magnitude = base_magnitude
        
        # Create scale-specific frequency ranges
        self.scale_perturbations = nn.ModuleList()
        for i in range(num_scales):
            # Divide frequency spectrum into bands
            low = i / num_scales
            high = (i + 1) / num_scales
            magnitude = base_magnitude / (i + 1)  # Lower magnitude for higher frequencies
            
            self.scale_perturbations.append(
                FFTPhasePerturbation(
                    image_size=image_size,
                    magnitude=magnitude,
                    learnable=learnable,
                    frequency_range=(low, high)
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale phase perturbation."""
        for perturbation in self.scale_perturbations:
            x = perturbation(x)
        return x
    
    def reset_parameters(self):
        """Reset all scale parameters."""
        for perturbation in self.scale_perturbations:
            perturbation.reset_parameters()