"""
Rolling Shutter Effect Module.

This module implements differentiable rolling shutter artifacts.
Rolling shutter is a camera sensor effect where different rows/columns
are exposed at different times, causing geometric distortions for
moving objects or camera motion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class RollingShutter(nn.Module):
    """
    Differentiable rolling shutter effect.
    
    Simulates the rolling shutter artifact where each row (or column)
    of the image is displaced based on a time-varying offset function.
    This creates realistic camera motion artifacts.
    
    Supports configurable number of harmonic modes for richer displacement
    patterns, and optional per-row amplitude for spatially varying effects.
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        max_offset: Maximum pixel offset for the rolling shutter effect.
        direction: Scan direction ('horizontal' or 'vertical').
        wave_frequency: Frequency of the displacement wave pattern.
        wave_type: Type of wave ('sine', 'linear', 'random').
        learnable: If True, wave parameters are learnable.
        num_harmonics: Number of harmonic modes (default: 2 for backward compat).
            More harmonics = more complex displacement patterns = more parameters.
        per_row_amplitude: If True, use a learnable amplitude per scan row/column
            instead of a single scalar amplitude, greatly increasing parameters.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        max_offset: float = 10.0,
        direction: str = 'horizontal',
        wave_frequency: float = 1.0,
        wave_type: str = 'sine',
        learnable: bool = True,
        num_harmonics: int = 2,
        per_row_amplitude: bool = False
    ):
        super().__init__()
        self.image_size = image_size
        self.max_offset = max_offset
        self.direction = direction
        self.wave_frequency = wave_frequency
        self.wave_type = wave_type
        self.learnable = learnable
        self.num_harmonics = num_harmonics
        self.per_row_amplitude = per_row_amplitude
        
        # Initialize wave parameters
        self._init_wave_parameters()
        
        # Create base coordinate grid
        self._create_base_grid()
    
    def _init_wave_parameters(self):
        """Initialize learnable wave parameters."""
        # Determine amplitude shape
        if self.per_row_amplitude:
            # One amplitude per scan line
            if self.direction == 'horizontal':
                amp_shape = (self.image_size[0],)
            else:
                amp_shape = (self.image_size[1],)
        else:
            amp_shape = (1,)
        
        if self.learnable:
            # Harmonic parameters: amplitude, frequency, phase for each harmonic
            self.amplitudes = nn.ParameterList()
            self.frequencies = nn.ParameterList()
            self.phases = nn.ParameterList()
            
            for i in range(self.num_harmonics):
                self.amplitudes.append(nn.Parameter(
                    torch.zeros(*amp_shape), requires_grad=True
                ))
                self.frequencies.append(nn.Parameter(
                    torch.tensor([self.wave_frequency * (i + 1)]),
                    requires_grad=True
                ))
                self.phases.append(nn.Parameter(
                    torch.zeros(1), requires_grad=True
                ))
        else:
            # Use buffers for non-learnable mode
            self._amplitudes_buffers = []
            self._frequencies_buffers = []
            self._phases_buffers = []
            for i in range(self.num_harmonics):
                self.register_buffer(f'amp_{i}', torch.zeros(*amp_shape))
                self.register_buffer(f'freq_{i}', torch.tensor([self.wave_frequency * (i + 1)]))
                self.register_buffer(f'phase_{i}', torch.zeros(1))
    
    def _create_base_grid(self):
        """Create base coordinate grid for sampling."""
        h, w = self.image_size
        
        # Create normalized coordinate grid [0, 1]
        if self.direction == 'horizontal':
            # Scan direction is horizontal (rows displaced)
            scan_coords = torch.linspace(0, 1, h)
            other_coords = torch.linspace(-1, 1, w)
            scan_grid, other_grid = torch.meshgrid(scan_coords, other_coords, indexing='ij')
        else:
            # Scan direction is vertical (columns displaced)
            scan_coords = torch.linspace(0, 1, w)
            other_coords = torch.linspace(-1, 1, h)
            other_grid, scan_grid = torch.meshgrid(other_coords, scan_coords, indexing='ij')
        
        self.register_buffer('scan_coords', scan_grid)
        self.register_buffer('other_coords', other_grid)
    
    def get_displacement(self) -> torch.Tensor:
        """
        Compute the displacement field for rolling shutter effect.
        
        Returns:
            Displacement tensor of shape (H, W)
        """
        displacement = torch.zeros_like(self.scan_coords)
        
        if self.learnable:
            # Sum over all harmonic modes
            for i in range(self.num_harmonics):
                amp = torch.tanh(self.amplitudes[i]) * self.max_offset / (i + 1)
                freq = torch.abs(self.frequencies[i]) + 0.1  # Ensure positive frequency
                phase = self.phases[i] * 2 * np.pi
                
                # Handle per-row amplitude: amp shape is (H,) or (W,)
                if self.per_row_amplitude:
                    # amp: (num_scan_lines,), need to broadcast with scan_coords (H, W)
                    if self.direction == 'horizontal':
                        # scan_coords shape: (H, W), amp shape: (H,)
                        amp_2d = amp.unsqueeze(-1)  # (H, 1)
                    else:
                        amp_2d = amp.unsqueeze(0)  # (1, W)
                    displacement = displacement + amp_2d * torch.sin(
                        2 * np.pi * freq * self.scan_coords + phase
                    )
                else:
                    displacement = displacement + amp * torch.sin(
                        2 * np.pi * freq * self.scan_coords + phase
                    )
        else:
            # Non-learnable mode: use buffers
            for i in range(self.num_harmonics):
                amp = torch.tanh(getattr(self, f'amp_{i}')) * self.max_offset / (i + 1)
                freq = torch.abs(getattr(self, f'freq_{i}')) + 0.1
                phase = getattr(self, f'phase_{i}') * 2 * np.pi
                
                if self.per_row_amplitude:
                    if self.direction == 'horizontal':
                        amp_2d = amp.unsqueeze(-1)
                    else:
                        amp_2d = amp.unsqueeze(0)
                    displacement = displacement + amp_2d * torch.sin(
                        2 * np.pi * freq * self.scan_coords + phase
                    )
                else:
                    displacement = displacement + amp * torch.sin(
                        2 * np.pi * freq * self.scan_coords + phase
                    )
        
        # Handle linear wave type (override harmonic sum)
        if self.wave_type == 'linear':
            if self.learnable:
                amp = torch.tanh(self.amplitudes[0]) * self.max_offset
            else:
                amp = torch.tanh(getattr(self, 'amp_0')) * self.max_offset
            displacement = amp * self.scan_coords
        
        return displacement
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rolling shutter effect to input images.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Transformed tensor of shape (B, C, H, W)
        """
        batch_size, channels, h, w = x.shape
        
        # Get displacement field
        displacement = self.get_displacement()
        
        # Create sampling grid
        grid = self._create_sampling_grid(displacement, h, w, x.device)
        
        # Apply grid sample
        output = F.grid_sample(
            x, grid.expand(batch_size, -1, -1, -1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        return output
    
    def _create_sampling_grid(
        self, 
        displacement: torch.Tensor, 
        h: int, 
        w: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Create sampling grid with rolling shutter displacement.
        
        Args:
            displacement: Displacement field (H, W)
            h, w: Image dimensions
            device: Device to create tensors on
            
        Returns:
            Sampling grid of shape (1, H, W, 2)
        """
        # Create base grid in normalized coordinates [-1, 1]
        y_coords = torch.linspace(-1, 1, h, device=device)
        x_coords = torch.linspace(-1, 1, w, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Normalize displacement to [-1, 1]
        if self.direction == 'horizontal':
            # Displace in x direction
            x_displaced = x_grid + displacement / (w / 2)
            y_displaced = y_grid
        else:
            # Displace in y direction
            x_displaced = x_grid
            y_displaced = y_grid + displacement / (h / 2)
        
        # Stack to create grid
        grid = torch.stack([x_displaced, y_displaced], dim=-1).unsqueeze(0)
        
        return grid
    
    def reset_parameters(self):
        """Reset wave parameters to zero."""
        if self.learnable:
            for i in range(self.num_harmonics):
                nn.init.zeros_(self.amplitudes[i])
                nn.init.constant_(self.frequencies[i], self.wave_frequency * (i + 1))
                nn.init.zeros_(self.phases[i])
    
    def randomize(self, magnitude: Optional[float] = None):
        """
        Randomize wave parameters.
        
        Args:
            magnitude: Optional magnitude override for randomization
        """
        mag = magnitude if magnitude is not None else self.max_offset
        
        if self.learnable:
            with torch.no_grad():
                for i in range(self.num_harmonics):
                    scale = 0.5 / (i + 1)
                    self.amplitudes[i].data = torch.randn_like(self.amplitudes[i]) * scale
                    self.frequencies[i].data = torch.abs(torch.randn_like(self.frequencies[i])) + 0.5
                    self.phases[i].data = torch.rand_like(self.phases[i]) * 2 - 1
    
    def get_offset_magnitude(self) -> float:
        """Get the current average offset magnitude."""
        displacement = self.get_displacement()
        return displacement.abs().mean().item()
    
    def extra_repr(self) -> str:
        return (f'image_size={self.image_size}, max_offset={self.max_offset}, '
                f'direction={self.direction}, wave_type={self.wave_type}')


class RollingShutterComplex(nn.Module):
    """
    Complex rolling shutter effect with multiple motion components.
    
    Simulates more realistic rolling shutter artifacts with:
    - Camera translation
    - Camera rotation
    - Vibration/wobble effects
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        max_translation: Maximum translation offset.
        max_rotation: Maximum rotation angle in radians.
        max_vibration: Maximum vibration amplitude.
        learnable: If True, all parameters are learnable.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        max_translation: float = 10.0,
        max_rotation: float = 0.05,
        max_vibration: float = 3.0,
        learnable: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.max_vibration = max_vibration
        
        # Translation parameters
        if learnable:
            self.tx = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.ty = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.register_buffer('tx', torch.zeros(1))
            self.register_buffer('ty', torch.zeros(1))
        
        # Rotation parameters
        if learnable:
            self.rotation = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.register_buffer('rotation', torch.zeros(1))
        
        # Vibration parameters
        if learnable:
            self.vib_amp = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.vib_freq = nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            self.register_buffer('vib_amp', torch.zeros(1))
            self.register_buffer('vib_freq', torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex rolling shutter effect."""
        batch_size, channels, h, w = x.shape
        
        # Create sampling grid
        grid = self._create_complex_grid(h, w, x.device)
        
        # Apply grid sample
        output = F.grid_sample(
            x, grid.expand(batch_size, -1, -1, -1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        return output
    
    def _create_complex_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Create complex sampling grid with all motion components."""
        # Base grid
        y_coords = torch.linspace(-1, 1, h, device=device)
        x_coords = torch.linspace(-1, 1, w, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Normalized scan position (0 to 1)
        scan_pos = (y_grid + 1) / 2
        
        # Translation component (linear with scan position)
        tx = torch.tanh(self.tx) * self.max_translation / (w / 2) * scan_pos
        ty = torch.tanh(self.ty) * self.max_translation / (h / 2) * scan_pos
        
        # Rotation component (increases with scan position)
        angle = torch.tanh(self.rotation) * self.max_rotation * scan_pos
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # Apply rotation
        x_rot = x_grid * cos_a - y_grid * sin_a
        y_rot = x_grid * sin_a + y_grid * cos_a
        
        # Vibration component (sinusoidal)
        vib_amp = torch.tanh(self.vib_amp) * self.max_vibration / (w / 2)
        vib_freq = torch.abs(self.vib_freq) + 0.1
        vibration = vib_amp * torch.sin(2 * np.pi * vib_freq * scan_pos)
        
        # Combine all components
        x_final = x_rot + tx + vibration
        y_final = y_rot + ty
        
        return torch.stack([x_final, y_final], dim=-1).unsqueeze(0)


class RollingShutterRealistic(nn.Module):
    """
    Realistic rolling shutter simulation.
    
    Models the actual sensor readout process with:
    - Row-by-row exposure timing
    - Motion blur during exposure
    - Variable scan speed
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        readout_time: Total sensor readout time in arbitrary units.
        motion_velocity: Camera motion velocity (pixels per unit time).
        motion_direction: Direction of motion ('horizontal', 'vertical', 'diagonal').
        learnable: If True, motion parameters are learnable.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        readout_time: float = 1.0,
        motion_velocity: float = 10.0,
        motion_direction: str = 'horizontal',
        learnable: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.readout_time = readout_time
        self.motion_velocity = motion_velocity
        self.motion_direction = motion_direction
        
        if learnable:
            self.velocity_x = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.velocity_y = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.register_buffer('velocity_x', torch.zeros(1))
            self.register_buffer('velocity_y', torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply realistic rolling shutter effect."""
        batch_size, channels, h, w = x.shape
        
        # Get motion velocities
        vx = torch.tanh(self.velocity_x) * self.motion_velocity
        vy = torch.tanh(self.velocity_y) * self.motion_velocity
        
        # Create sampling grid
        grid = self._create_realistic_grid(vx, vy, h, w, x.device)
        
        # Apply grid sample
        output = F.grid_sample(
            x, grid.expand(batch_size, -1, -1, -1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        return output
    
    def _create_realistic_grid(
        self, 
        vx: torch.Tensor, 
        vy: torch.Tensor, 
        h: int, 
        w: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Create realistic rolling shutter grid."""
        # Base grid
        y_coords = torch.linspace(-1, 1, h, device=device)
        x_coords = torch.linspace(-1, 1, w, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Time for each row (normalized 0 to 1)
        row_time = (y_grid + 1) / 2
        
        # Displacement based on motion velocity and row time
        dx = vx * row_time / (w / 2)
        dy = vy * row_time / (h / 2)
        
        # Apply displacement
        x_final = x_grid + dx
        y_final = y_grid + dy
        
        return torch.stack([x_final, y_final], dim=-1).unsqueeze(0)


class RollingShutterWobble(nn.Module):
    """
    Rolling shutter with wobble/vibration effect.
    
    Simulates high-frequency vibrations during sensor readout,
    common in cameras with mechanical stabilization issues.
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        max_wobble: Maximum wobble amplitude in pixels.
        wobble_frequency: Frequency of wobble oscillation.
        num_modes: Number of vibration modes to simulate.
        learnable: If True, wobble parameters are learnable.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        max_wobble: float = 5.0,
        wobble_frequency: float = 10.0,
        num_modes: int = 3,
        learnable: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.max_wobble = max_wobble
        self.num_modes = num_modes
        
        # Initialize wobble modes
        if learnable:
            self.amplitudes = nn.Parameter(
                torch.zeros(num_modes, 2),  # (num_modes, 2) for x and y
                requires_grad=True
            )
            self.frequencies = nn.Parameter(
                torch.ones(num_modes, 2) * wobble_frequency,
                requires_grad=True
            )
            self.phases = nn.Parameter(
                torch.zeros(num_modes, 2),
                requires_grad=True
            )
        else:
            self.register_buffer('amplitudes', torch.zeros(num_modes, 2))
            self.register_buffer('frequencies', torch.ones(num_modes, 2) * wobble_frequency)
            self.register_buffer('phases', torch.zeros(num_modes, 2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply wobble rolling shutter effect."""
        batch_size, channels, h, w = x.shape
        
        # Create sampling grid
        grid = self._create_wobble_grid(h, w, x.device)
        
        # Apply grid sample
        output = F.grid_sample(
            x, grid.expand(batch_size, -1, -1, -1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        return output
    
    def _create_wobble_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Create wobble sampling grid."""
        # Base grid
        y_coords = torch.linspace(-1, 1, h, device=device)
        x_coords = torch.linspace(-1, 1, w, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Row time (0 to 1)
        row_time = (y_grid + 1) / 2
        
        # Compute wobble displacement
        dx = torch.zeros_like(x_grid)
        dy = torch.zeros_like(y_grid)
        
        for i in range(self.num_modes):
            amp = torch.tanh(self.amplitudes[i]) * self.max_wobble
            freq = torch.abs(self.frequencies[i])
            phase = self.phases[i] * 2 * np.pi
            
            dx = dx + amp[0] * torch.sin(2 * np.pi * freq[0] * row_time + phase[0])
            dy = dy + amp[1] * torch.sin(2 * np.pi * freq[1] * row_time + phase[1])
        
        # Normalize displacement
        dx = dx / (w / 2)
        dy = dy / (h / 2)
        
        # Apply displacement
        x_final = x_grid + dx
        y_final = y_grid + dy
        
        return torch.stack([x_final, y_final], dim=-1).unsqueeze(0)