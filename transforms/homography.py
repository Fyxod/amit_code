"""
Homography Transformation Module.

This module implements differentiable homography (perspective) transformations.
A homography is an 8-DOF projective transformation that can model perspective
changes, rotation, translation, and scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class HomographyTransform(nn.Module):
    """
    Differentiable homography transformation.
    
    Applies a learnable perspective transformation to the input image.
    The homography matrix has 8 degrees of freedom (9 elements with
    constraint h[2,2] = 1).
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        max_perturbation: Maximum perturbation for homography parameters.
        learnable: If True, homography parameters are learnable.
        preserve_aspect_ratio: If True, constrain to similarity transform.
        init_method: Initialization method ('identity', 'random', 'perspective').
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        max_perturbation: float = 0.1,
        learnable: bool = True,
        preserve_aspect_ratio: bool = False,
        init_method: str = 'identity'
    ):
        super().__init__()
        self.image_size = image_size
        self.max_perturbation = max_perturbation
        self.learnable = learnable
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.init_method = init_method
        
        # Initialize homography parameters
        self._init_homography_params()
        
        # Create base coordinate grid
        self._create_base_grid()
    
    def _init_homography_params(self):
        """Initialize homography parameters."""
        if self.preserve_aspect_ratio:
            # Similarity transform: rotation, scale, translation (4 DOF)
            if self.learnable:
                self.theta = nn.Parameter(torch.zeros(1), requires_grad=True)  # Rotation
                self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)  # Log scale
                self.tx = nn.Parameter(torch.zeros(1), requires_grad=True)  # Translation x
                self.ty = nn.Parameter(torch.zeros(1), requires_grad=True)  # Translation y
            else:
                self.register_buffer('theta', torch.zeros(1))
                self.register_buffer('scale', torch.zeros(1))
                self.register_buffer('tx', torch.zeros(1))
                self.register_buffer('ty', torch.zeros(1))
        else:
            # Full homography: 8 DOF (excluding h[2,2] which is fixed to 1)
            if self.learnable:
                # Parameterize as perturbation from identity
                self.homography_delta = nn.Parameter(
                    torch.zeros(8),
                    requires_grad=True
                )
            else:
                self.register_buffer('homography_delta', torch.zeros(8))
    
    def _create_base_grid(self):
        """Create base coordinate grid for sampling."""
        h, w = self.image_size
        
        # Create normalized coordinate grid [-1, 1]
        y_coords = torch.linspace(-1, 1, h)
        x_coords = torch.linspace(-1, 1, w)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack to create grid of shape (H, W, 2)
        base_grid = torch.stack([x_grid, y_grid], dim=-1)
        
        self.register_buffer('base_grid', base_grid)
    
    def get_homography_matrix(self) -> torch.Tensor:
        """
        Get the current homography matrix.
        
        Returns:
            Homography matrix of shape (3, 3)
        """
        if self.preserve_aspect_ratio:
            # Build similarity transform matrix
            theta = torch.tanh(self.theta) * np.pi  # Rotation in [-pi, pi]
            s = torch.exp(torch.tanh(self.scale) * 0.5)  # Scale in [e^-0.5, e^0.5]
            tx = torch.tanh(self.tx) * self.max_perturbation * 2
            ty = torch.tanh(self.ty) * self.max_perturbation * 2
            
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            
            H = torch.tensor([
                [s * cos_t, -s * sin_t, tx],
                [s * sin_t, s * cos_t, ty],
                [0, 0, 1]
            ], device=theta.device, dtype=theta.dtype)
        else:
            # Build full homography matrix
            # Start with identity
            H = torch.eye(3, device=self.homography_delta.device, dtype=self.homography_delta.dtype)
            
            # Add perturbation (bounded by tanh)
            delta = torch.tanh(self.homography_delta) * self.max_perturbation
            
            # Fill in the 8 parameters (h[2,2] = 1)
            H[0, 0] = 1 + delta[0]  # h11
            H[0, 1] = delta[1]      # h12
            H[0, 2] = delta[2]      # h13
            H[1, 0] = delta[3]      # h21
            H[1, 1] = 1 + delta[4]  # h22
            H[1, 2] = delta[5]      # h23
            H[2, 0] = delta[6]      # h31
            H[2, 1] = delta[7]      # h32
            # H[2, 2] = 1 (already set)
        
        return H
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply homography transformation to input images.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Transformed tensor of shape (B, C, H, W)
        """
        batch_size = x.shape[0]
        h, w = self.image_size
        
        # Get homography matrix
        H = self.get_homography_matrix()
        
        # Create sampling grid
        grid = self._create_sampling_grid(H, batch_size, h, w)
        
        # Apply grid sample
        output = F.grid_sample(
            x, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        return output
    
    def _create_sampling_grid(
        self, 
        H: torch.Tensor, 
        batch_size: int, 
        h: int, 
        w: int
    ) -> torch.Tensor:
        """
        Create sampling grid using homography matrix.
        
        Args:
            H: Homography matrix (3, 3)
            batch_size: Number of images in batch
            h, w: Image dimensions
            
        Returns:
            Sampling grid of shape (B, H, W, 2)
        """
        # Flatten base grid to (H*W, 2)
        flat_grid = self.base_grid.view(-1, 2)
        
        # Convert to homogeneous coordinates (H*W, 3)
        ones = torch.ones(flat_grid.shape[0], 1, device=flat_grid.device)
        homogeneous = torch.cat([flat_grid, ones], dim=1)
        
        # Apply inverse homography (to get source coordinates)
        H_inv = torch.inverse(H)
        transformed = torch.matmul(homogeneous, H_inv.T)
        
        # Convert back to Cartesian coordinates
        transformed = transformed[:, :2] / (transformed[:, 2:3] + 1e-8)
        
        # Reshape to grid
        grid = transformed.view(h, w, 2)
        
        # Expand for batch
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return grid
    
    def reset_parameters(self):
        """Reset homography parameters to identity."""
        if self.learnable:
            if self.preserve_aspect_ratio:
                nn.init.zeros_(self.theta)
                nn.init.zeros_(self.scale)
                nn.init.zeros_(self.tx)
                nn.init.zeros_(self.ty)
            else:
                nn.init.zeros_(self.homography_delta)
    
    def randomize(self, magnitude: Optional[float] = None):
        """
        Randomize homography parameters.
        
        Args:
            magnitude: Optional magnitude override for randomization
        """
        mag = magnitude if magnitude is not None else self.max_perturbation
        
        if self.learnable:
            with torch.no_grad():
                if self.preserve_aspect_ratio:
                    self.theta.data = torch.randn_like(self.theta) * 0.5
                    self.scale.data = torch.randn_like(self.scale) * 0.5
                    self.tx.data = torch.randn_like(self.tx) * 0.5
                    self.ty.data = torch.randn_like(self.ty) * 0.5
                else:
                    self.homography_delta.data = torch.randn_like(self.homography_delta) * 0.5
        else:
            if self.preserve_aspect_ratio:
                self.theta = torch.randn_like(self.theta) * 0.5
                self.scale = torch.randn_like(self.scale) * 0.5
                self.tx = torch.randn_like(self.tx) * 0.5
                self.ty = torch.randn_like(self.ty) * 0.5
            else:
                self.homography_delta = torch.randn_like(self.homography_delta) * 0.5
    
    def get_perturbation_magnitude(self) -> float:
        """Get the current magnitude of homography perturbation."""
        if self.preserve_aspect_ratio:
            return torch.sqrt(
                self.theta ** 2 + self.scale ** 2 + 
                self.tx ** 2 + self.ty ** 2
            ).item()
        else:
            return torch.norm(torch.tanh(self.homography_delta) * self.max_perturbation).item()
    
    def extra_repr(self) -> str:
        return (f'image_size={self.image_size}, max_perturbation={self.max_perturbation}, '
                f'learnable={self.learnable}, preserve_aspect_ratio={self.preserve_aspect_ratio}')


class RandomHomography(nn.Module):
    """
    Random homography transformation for data augmentation.
    
    Applies random perspective transformations within specified bounds.
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        rotation_range: Maximum rotation angle in degrees.
        scale_range: Tuple of (min_scale, max_scale).
        translation_range: Maximum translation as fraction of image size.
        perspective_range: Maximum perspective distortion.
        shear_range: Maximum shear angle in degrees.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        rotation_range: float = 15.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translation_range: float = 0.1,
        perspective_range: float = 0.1,
        shear_range: float = 10.0
    ):
        super().__init__()
        self.image_size = image_size
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.perspective_range = perspective_range
        self.shear_range = shear_range
        
        # Create base grid
        h, w = image_size
        y_coords = torch.linspace(-1, 1, h)
        x_coords = torch.linspace(-1, 1, w)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        base_grid = torch.stack([x_grid, y_grid], dim=-1)
        self.register_buffer('base_grid', base_grid)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random homography transformation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Transformed tensor of shape (B, C, H, W)
        """
        batch_size = x.shape[0]
        h, w = self.image_size
        
        # Generate random homography parameters
        H = self._sample_random_homography(batch_size, x.device)
        
        # Create sampling grid
        grid = self._create_sampling_grid(H, batch_size, h, w)
        
        # Apply grid sample
        output = F.grid_sample(
            x, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        return output
    
    def _sample_random_homography(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random homography matrix."""
        # Random rotation
        angle = (torch.rand(batch_size, device=device) - 0.5) * 2 * np.radians(self.rotation_range)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # Random scale
        scale = self.scale_range[0] + torch.rand(batch_size, device=device) * (self.scale_range[1] - self.scale_range[0])
        
        # Random translation
        tx = (torch.rand(batch_size, device=device) - 0.5) * 2 * self.translation_range
        ty = (torch.rand(batch_size, device=device) - 0.5) * 2 * self.translation_range
        
        # Random perspective
        px = (torch.rand(batch_size, device=device) - 0.5) * 2 * self.perspective_range
        py = (torch.rand(batch_size, device=device) - 0.5) * 2 * self.perspective_range
        
        # Random shear
        shear_x = (torch.rand(batch_size, device=device) - 0.5) * 2 * np.radians(self.shear_range)
        shear_y = (torch.rand(batch_size, device=device) - 0.5) * 2 * np.radians(self.shear_range)
        
        # Build homography matrix
        H = torch.zeros(batch_size, 3, 3, device=device)
        
        # Rotation + Scale
        H[:, 0, 0] = scale * cos_a
        H[:, 0, 1] = -scale * sin_a
        H[:, 1, 0] = scale * sin_a
        H[:, 1, 1] = scale * cos_a
        
        # Translation
        H[:, 0, 2] = tx
        H[:, 1, 2] = ty
        
        # Shear
        H[:, 0, 0] += H[:, 1, 0] * torch.tan(shear_x)
        H[:, 1, 1] += H[:, 0, 1] * torch.tan(shear_y)
        
        # Perspective
        H[:, 2, 0] = px
        H[:, 2, 1] = py
        H[:, 2, 2] = 1
        
        return H
    
    def _create_sampling_grid(
        self, 
        H: torch.Tensor, 
        batch_size: int, 
        h: int, 
        w: int
    ) -> torch.Tensor:
        """Create sampling grid using homography matrices."""
        # Flatten base grid
        flat_grid = self.base_grid.view(-1, 2)
        ones = torch.ones(flat_grid.shape[0], 1, device=flat_grid.device)
        homogeneous = torch.cat([flat_grid, ones], dim=1)
        
        # Apply inverse homography for each batch
        grids = []
        for i in range(batch_size):
            H_inv = torch.inverse(H[i])
            transformed = torch.matmul(homogeneous, H_inv.T)
            transformed = transformed[:, :2] / (transformed[:, 2:3] + 1e-8)
            grids.append(transformed.view(h, w, 2))
        
        return torch.stack(grids)


class HomographyFlow(nn.Module):
    """
    Homography-based dense flow field.
    
    Combines homography transformation with local flow perturbations
    for more flexible geometric transformations.
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        homography_perturbation: Maximum homography perturbation.
        flow_perturbation: Maximum local flow perturbation.
        flow_grid_size: Size of the flow control grid.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        homography_perturbation: float = 0.1,
        flow_perturbation: float = 5.0,
        flow_grid_size: int = 8
    ):
        super().__init__()
        self.homography = HomographyTransform(
            image_size=image_size,
            max_perturbation=homography_perturbation,
            learnable=True
        )
        
        self.flow_grid_size = flow_grid_size
        self.flow_perturbation = flow_perturbation
        
        # Learnable flow field on a coarse grid
        self.flow_field = nn.Parameter(
            torch.zeros(2, flow_grid_size, flow_grid_size),
            requires_grad=True
        )
        
        self.image_size = image_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply homography + flow transformation."""
        batch_size, channels, h, w = x.shape
        
        # Apply homography first
        x_homography = self.homography(x)
        
        # Interpolate flow field to full resolution
        flow = F.interpolate(
            self.flow_field.unsqueeze(0),
            size=(h, w),
            mode='bilinear',
            align_corners=True
        ).squeeze(0)
        
        # Bound flow
        flow = torch.tanh(flow) * self.flow_perturbation
        
        # Create sampling grid
        y_coords = torch.linspace(-1, 1, h, device=x.device)
        x_coords = torch.linspace(-1, 1, w, device=x.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        base_grid = torch.stack([x_grid, y_grid], dim=-1)
        
        # Add flow to grid (normalized)
        flow_norm = flow.permute(1, 2, 0)
        flow_norm[..., 0] = flow_norm[..., 0] / (w / 2)
        flow_norm[..., 1] = flow_norm[..., 1] / (h / 2)
        
        grid = base_grid + flow_norm
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Apply flow
        output = F.grid_sample(
            x_homography, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        return output


class PiecewiseHomography(nn.Module):
    """
    Piecewise homography transformation.
    
    Divides the image into a grid of patches and applies a separate
    learnable homography to each patch, then stitches the results back
    together. This provides significantly more degrees of freedom than
    a single global homography.
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        grid_size: Tuple of (rows, cols) for the patch grid.
        max_perturbation: Maximum perturbation for each homography parameter.
        learnable: If True, homography parameters are learnable.
        blend_margin: Width of blending margin between patches (in pixels).
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        grid_size: Tuple[int, int] = (4, 4),
        max_perturbation: float = 0.1,
        learnable: bool = True,
        blend_margin: int = 8
    ):
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        self.max_perturbation = max_perturbation
        self.learnable = learnable
        self.blend_margin = blend_margin
        
        self.num_patches = grid_size[0] * grid_size[1]
        
        # Initialize per-patch homography parameters
        if learnable:
            # Each patch has 8 homography parameters
            self.homography_deltas = nn.Parameter(
                torch.zeros(self.num_patches, 8),
                requires_grad=True
            )
        else:
            self.register_buffer(
                'homography_deltas',
                torch.zeros(self.num_patches, 8)
            )
        
        # Create base coordinate grid
        self._create_base_grid()
        
        # Create blending weights for smooth patch boundaries
        self._create_blend_weights()
    
    def _create_base_grid(self):
        """Create base coordinate grid for sampling."""
        h, w = self.image_size
        
        y_coords = torch.linspace(-1, 1, h)
        x_coords = torch.linspace(-1, 1, w)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        base_grid = torch.stack([x_grid, y_grid], dim=-1)
        self.register_buffer('base_grid', base_grid)
    
    def _create_blend_weights(self):
        """Create blending weights for smooth transitions between patches."""
        h, w = self.image_size
        rows, cols = self.grid_size
        
        patch_h = h // rows
        patch_w = w // cols
        margin = self.blend_margin
        
        # Create weight map (1 everywhere, with smooth transitions at boundaries)
        weight_map = torch.ones(1, 1, h, w)
        
        for r in range(rows):
            for c in range(cols):
                y_start = r * patch_h
                y_end = min((r + 1) * patch_h, h)
                x_start = c * patch_w
                x_end = min((c + 1) * patch_w, w)
                
                # Apply smooth falloff at patch boundaries
                if r > 0:
                    for y in range(y_start, min(y_start + margin, y_end)):
                        t = (y - y_start) / max(margin, 1)
                        weight_map[0, 0, y, x_start:x_end] = torch.minimum(
                            weight_map[0, 0, y, x_start:x_end],
                            torch.tensor(t)
                        )
                if r < rows - 1:
                    for y in range(max(y_end - margin, y_start), y_end):
                        t = (y_end - y) / max(margin, 1)
                        weight_map[0, 0, y, x_start:x_end] = torch.minimum(
                            weight_map[0, 0, y, x_start:x_end],
                            torch.tensor(t)
                        )
                if c > 0:
                    for x in range(x_start, min(x_start + margin, x_end)):
                        t = (x - x_start) / max(margin, 1)
                        weight_map[0, 0, y_start:y_end, x] = torch.minimum(
                            weight_map[0, 0, y_start:y_end, x],
                            torch.tensor(t)
                        )
                if c < cols - 1:
                    for x in range(max(x_end - margin, x_start), x_end):
                        t = (x_end - x) / max(margin, 1)
                        weight_map[0, 0, y_start:y_end, x] = torch.minimum(
                            weight_map[0, 0, y_start:y_end, x],
                            torch.tensor(t)
                        )
        
        self.register_buffer('blend_weights', weight_map)
    
    def get_homography_matrices(self) -> torch.Tensor:
        """
        Get homography matrices for all patches.
        
        Returns:
            Homography matrices of shape (num_patches, 3, 3)
        """
        # Build homography matrices from bounded deltas
        delta = torch.tanh(self.homography_deltas) * self.max_perturbation
        
        # Create identity matrices for all patches
        H = torch.eye(3, device=delta.device, dtype=delta.dtype)
        H = H.unsqueeze(0).expand(self.num_patches, -1, -1).clone()
        
        # Fill in the 8 parameters per patch
        H[:, 0, 0] = 1 + delta[:, 0]
        H[:, 0, 1] = delta[:, 1]
        H[:, 0, 2] = delta[:, 2]
        H[:, 1, 0] = delta[:, 3]
        H[:, 1, 1] = 1 + delta[:, 4]
        H[:, 1, 2] = delta[:, 5]
        H[:, 2, 0] = delta[:, 6]
        H[:, 2, 1] = delta[:, 7]
        
        return H
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply piecewise homography transformation to input images.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Transformed tensor of shape (B, C, H, W)
        """
        batch_size, channels, h, w = x.shape
        rows, cols = self.grid_size
        patch_h = h // rows
        patch_w = w // cols
        
        # Get all homography matrices
        H_all = self.get_homography_matrices()  # (num_patches, 3, 3)
        
        # Process each patch
        output = torch.zeros_like(x)
        weight_sum = torch.zeros(batch_size, 1, h, w, device=x.device)
        
        for idx in range(self.num_patches):
            r = idx // cols
            c = idx % cols
            
            # Patch boundaries
            y_start = r * patch_h
            y_end = min((r + 1) * patch_h, h)
            x_start = c * patch_w
            x_end = min((c + 1) * patch_w, w)
            
            # Get patch grid coordinates
            patch_grid = self.base_grid[y_start:y_end, x_start:x_end]  # (ph, pw, 2)
            
            # Apply inverse homography for this patch
            H_inv = torch.inverse(H_all[idx])  # (3, 3)
            
            flat_grid = patch_grid.reshape(-1, 2)
            ones = torch.ones(flat_grid.shape[0], 1, device=flat_grid.device)
            homogeneous = torch.cat([flat_grid, ones], dim=1)
            
            transformed = torch.matmul(homogeneous, H_inv.T)
            transformed = transformed[:, :2] / (transformed[:, 2:3] + 1e-8)
            
            grid = transformed.reshape(y_end - y_start, x_end - x_start, 2)
            grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
            
            # Sample from the full input image
            patch_output = F.grid_sample(
                x, grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            
            # Add to output with blending
            output[:, :, y_start:y_end, x_start:x_end] += patch_output
            weight_sum[:, :, y_start:y_end, x_start:x_end] += 1.0
        
        # Normalize by weight sum (for blending at boundaries)
        output = output / (weight_sum + 1e-8)
        
        return output
    
    def reset_parameters(self):
        """Reset homography parameters to identity."""
        if self.learnable:
            nn.init.zeros_(self.homography_deltas)
    
    def get_perturbation_magnitude(self) -> float:
        """Get the current magnitude of homography perturbation."""
        return torch.norm(torch.tanh(self.homography_deltas) * self.max_perturbation).item()
    
    def extra_repr(self) -> str:
        return (f'image_size={self.image_size}, grid_size={self.grid_size}, '
                f'max_perturbation={self.max_perturbation}, learnable={self.learnable}')
