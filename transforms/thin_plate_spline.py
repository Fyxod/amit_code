"""
Thin-Plate Spline (TPS) Warp Module (Optimized).

This module implements differentiable thin-plate spline warping.
TPS is a non-rigid transformation that provides smooth interpolation
between control points, making it ideal for modeling complex geometric
deformations.

OPTIMIZED VERSION:
- Pre-computes TPS coefficient matrix once during initialization
- Caches radial basis values for all pixels
- Only updates displacement parameters during forward pass
- Vectorized grid computation instead of Python loops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class ThinPlateSpline(nn.Module):
    """
    Optimized Differentiable Thin-Plate Spline warp.
    
    TPS provides smooth non-rigid transformations by interpolating
    between control point displacements. The transformation minimizes
    bending energy while matching the control point constraints.
    
    OPTIMIZATION: Pre-computes all static matrices and caches
    radial basis values for fast forward passes.
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        num_control_points: Number of control points (will be arranged in a grid).
        max_displacement: Maximum pixel displacement for control points.
        learnable: If True, control point displacements are learnable.
        regularization: Regularization parameter for TPS fitting.
        border_mode: Border handling mode ('constant', 'reflect', 'replicate').
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        num_control_points: int = 16,
        max_displacement: float = 15.0,
        learnable: bool = True,
        regularization: float = 0.0,
        border_mode: str = 'constant'
    ):
        super().__init__()
        self.image_size = image_size
        self.num_control_points = num_control_points
        self.max_displacement = max_displacement
        self.learnable = learnable
        self.regularization = regularization
        self.border_mode = border_mode
        
        # Initialize control points (done once)
        self._init_control_points()
        
        # Initialize displacement parameters
        self._init_displacements()
        
        # Pre-compute TPS coefficient matrix (done once)
        self._compute_tps_coefficients()
        
        # Pre-compute radial basis values for all pixels (key optimization)
        self._precompute_radial_basis()
    
    def _init_control_points(self):
        """Initialize control points on a grid."""
        h, w = self.image_size
        
        # Calculate grid dimensions
        n = int(np.sqrt(self.num_control_points))
        
        # Create grid of control points (normalized to [-1, 1])
        y_coords = np.linspace(-1, 1, n)
        x_coords = np.linspace(-1, 1, n)
        
        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([x, y])
        
        control_points = np.array(points, dtype=np.float32)
        self.num_control_points = len(control_points)
        
        # Register control points as buffer
        self.register_buffer(
            'control_points',
            torch.from_numpy(control_points)
        )
        
        # Add corner points for boundary constraints
        corners = np.array([
            [-1, -1], [1, -1], [-1, 1], [1, 1]
        ], dtype=np.float32)
        
        self.register_buffer(
            'corner_points',
            torch.from_numpy(corners)
        )
        
        # All points (control + corners)
        all_points = np.vstack([control_points, corners])
        self.num_all_points = len(all_points)
        
        self.register_buffer(
            'all_points',
            torch.from_numpy(all_points)
        )
    
    def _init_displacements(self):
        """Initialize learnable displacement parameters."""
        if self.learnable:
            # Displacement for each control point (dx, dy)
            self.displacement = nn.Parameter(
                torch.zeros(self.num_control_points, 2),
                requires_grad=True
            )
        else:
            self.register_buffer(
                'displacement',
                torch.zeros(self.num_control_points, 2)
            )
        
        # Corner displacements are fixed to zero (boundary constraint)
        self.register_buffer(
            'corner_displacement',
            torch.zeros(4, 2)
        )
    
    def _compute_tps_coefficients(self):
        """Pre-compute TPS coefficient matrix (done once)."""
        # Get all points
        points = self.all_points  # (N, 2)
        n = self.num_all_points
        
        # Compute distance matrix
        dist_matrix = self._compute_distance_matrix(points)
        
        # Compute U matrix (radial basis function)
        U = self._radial_basis(dist_matrix)
        
        # Build the TPS coefficient matrix
        # [K  P] [w]   [v]
        # [P' 0] [a] = [0]
        # where K is the U matrix, P is the affine part
        
        # P matrix: [1, x, y] for each point
        P = torch.ones(n, 3, device=points.device, dtype=points.dtype)
        P[:, 1] = points[:, 0]  # x
        P[:, 2] = points[:, 1]  # y
        
        # Build full matrix with regularization to avoid singularity
        reg = max(self.regularization, 1e-5)  # Ensure minimum regularization
        K = U + reg * torch.eye(n, device=points.device, dtype=points.dtype)
        
        # Upper part: [K | P]
        upper = torch.cat([K, P], dim=1)
        
        # Lower part: [P' | 0]
        lower = torch.cat([P.T, torch.zeros(3, 3, device=points.device, dtype=points.dtype)], dim=1)
        
        # Full matrix
        L = torch.cat([upper, lower], dim=0)
        
        # Register L matrix (static)
        self.register_buffer('L_matrix', L)
        
        # Compute inverse (for solving the system) - done once!
        L_inv = torch.inverse(L)
        self.register_buffer('L_inverse', L_inv)
    
    def _precompute_radial_basis(self):
        """
        Pre-compute radial basis values for all pixels (key optimization).
        
        This avoids recomputing distances and radial basis values
        in every forward pass.
        """
        h, w = self.image_size
        device = self.all_points.device
        dtype = self.all_points.dtype
        
        # Create normalized coordinate grid [-1, 1]
        y_coords = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x_coords = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Flatten grid: (H*W, 2)
        grid_flat = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
        self.register_buffer('grid_flat', grid_flat)
        
        # Pre-compute distances from each pixel to all control points
        # grid_flat: (H*W, 2), all_points: (N, 2)
        # diff: (H*W, N, 2)
        diff = grid_flat.unsqueeze(1) - self.all_points.unsqueeze(0)
        
        # dist: (H*W, N)
        dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)
        
        # Pre-compute radial basis values: (H*W, N)
        U = self._radial_basis(dist)
        self.register_buffer('radial_basis_values', U)
        
        # Pre-compute affine part: [1, x, y] for each pixel
        affine = torch.ones(h * w, 3, device=device, dtype=dtype)
        affine[:, 1] = grid_flat[:, 0]  # x
        affine[:, 2] = grid_flat[:, 1]  # y
        self.register_buffer('affine_part', affine)
        
        # Pre-compute combined matrix: [U | affine] - (H*W, N+3)
        combined = torch.cat([U, affine], dim=1)
        self.register_buffer('combined_matrix', combined)
    
    def _compute_distance_matrix(self, points: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix."""
        # points: (N, 2)
        # Compute squared distances
        diff = points.unsqueeze(0) - points.unsqueeze(1)  # (N, N, 2)
        dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)  # (N, N)
        return dist
    
    def _radial_basis(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute TPS radial basis function.
        
        Uses U(r) = r^2 * log(r) for 2D TPS.
        """
        # Avoid log(0) by adding small epsilon
        r_safe = r + 1e-8
        return r_safe ** 2 * torch.log(r_safe)
    
    def get_displaced_points(self) -> torch.Tensor:
        """
        Get control points with applied displacements.
        
        Returns:
            Displaced control points tensor of shape (N, 2)
        """
        # Apply tanh to bound displacement
        bounded_displacement = torch.tanh(self.displacement) * self.max_displacement / self.image_size[0]
        
        # Combine control and corner displacements
        all_displacements = torch.cat([bounded_displacement, self.corner_displacement], dim=0)
        
        # Add displacement to control points
        displaced = self.all_points + all_displacements
        
        return displaced
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply TPS warp to input images (optimized - uses precomputed values).
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Warped tensor of shape (B, C, H, W)
        """
        batch_size, channels, h, w = x.shape
        device = x.device
        
        # Get displaced control points
        displaced_points = self.get_displaced_points()
        
        # Compute TPS transformation coefficients (fast matrix multiply)
        coeffs = self._solve_tps(displaced_points)
        
        # Create sampling grid using precomputed values (fast)
        grid = self._create_sampling_grid_fast(coeffs, h, w, device)
        
        # Apply grid sample
        output = F.grid_sample(
            x, grid.expand(batch_size, -1, -1, -1),
            mode='bilinear',
            padding_mode='zeros' if self.border_mode == 'constant' else self.border_mode,
            align_corners=True
        )
        
        return output
    
    def _solve_tps(self, target_points: torch.Tensor) -> torch.Tensor:
        """
        Solve TPS system to get transformation coefficients (fast).
        
        Uses precomputed L_inverse for fast matrix multiplication.
        
        Args:
            target_points: Target positions for all control points (N, 2)
            
        Returns:
            TPS coefficients for x and y transformations
        """
        # Build right-hand side
        n = self.num_all_points
        
        # Target positions (with zeros for the affine constraints)
        rhs_x = torch.zeros(n + 3, device=target_points.device, dtype=target_points.dtype)
        rhs_y = torch.zeros(n + 3, device=target_points.device, dtype=target_points.dtype)
        
        rhs_x[:n] = target_points[:, 0]
        rhs_y[:n] = target_points[:, 1]
        
        # Solve the system using precomputed inverse (fast!)
        coeffs_x = torch.matmul(self.L_inverse, rhs_x)
        coeffs_y = torch.matmul(self.L_inverse, rhs_y)
        
        return torch.stack([coeffs_x, coeffs_y], dim=0)
    
    def _create_sampling_grid_fast(
        self, 
        coeffs: torch.Tensor, 
        h: int, 
        w: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create sampling grid using precomputed values (fast).
        
        Uses precomputed radial basis values and affine part
        for fast transformation computation.
        
        Args:
            coeffs: TPS coefficients (2, N+3)
            h, w: Image dimensions
            device: Device
            
        Returns:
            Sampling grid of shape (1, H, W, 2)
        """
        # Use precomputed combined matrix: (H*W, N+3)
        # Apply coefficients: (H*W, N+3) @ (N+3, 2) -> (H*W, 2)
        transformed = torch.matmul(self.combined_matrix, coeffs.T)
        
        # Reshape to grid: (H, W, 2)
        grid = transformed.view(h, w, 2).unsqueeze(0)
        
        return grid
    
    def reset_parameters(self):
        """Reset displacement parameters to zero."""
        if self.learnable:
            nn.init.zeros_(self.displacement)
    
    def randomize(self, magnitude: Optional[float] = None):
        """
        Randomize displacement parameters.
        
        Args:
            magnitude: Optional magnitude override for randomization
        """
        mag = magnitude if magnitude is not None else self.max_displacement
        
        if self.learnable:
            with torch.no_grad():
                self.displacement.data = torch.randn_like(self.displacement) * 0.5
        else:
            self.displacement = torch.randn_like(self.displacement) * 0.5
    
    def get_displacement_magnitude(self) -> float:
        """Get the current average displacement magnitude."""
        bounded_displacement = torch.tanh(self.displacement) * self.max_displacement
        return torch.norm(bounded_displacement, dim=1).mean().item()
    
    def extra_repr(self) -> str:
        return (f'image_size={self.image_size}, num_control_points={self.num_control_points}, '
                f'max_displacement={self.max_displacement}, learnable={self.learnable}')


class ThinPlateSplineGrid(nn.Module):
    """
    TPS warp with grid-based control point placement (Optimized).
    
    Provides more control over the density and placement of control points.
    Uses precomputed matrices for fast forward passes.
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        grid_size: Tuple of (rows, cols) for control point grid.
        max_displacement: Maximum pixel displacement for control points.
        learnable: If True, control point displacements are learnable.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        grid_size: Tuple[int, int] = (4, 4),
        max_displacement: float = 15.0,
        learnable: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        self.max_displacement = max_displacement
        
        rows, cols = grid_size
        self.num_control_points = rows * cols
        
        # Initialize control points
        y_coords = torch.linspace(-1, 1, rows)
        x_coords = torch.linspace(-1, 1, cols)
        
        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([x, y])
        
        control_points = torch.stack([torch.tensor(p) for p in points])
        self.register_buffer('control_points', control_points)
        
        # Initialize displacements
        if learnable:
            self.displacement = nn.Parameter(
                torch.zeros(self.num_control_points, 2),
                requires_grad=True
            )
        else:
            self.register_buffer('displacement', torch.zeros(self.num_control_points, 2))
        
        # Pre-compute TPS matrices (done once)
        self._precompute_tps_matrices()
        
        # Pre-compute radial basis values (key optimization)
        self._precompute_radial_basis()
    
    def _precompute_tps_matrices(self):
        """Pre-compute TPS coefficient matrices (done once)."""
        points = self.control_points
        n = self.num_control_points
        
        # Distance matrix
        diff = points.unsqueeze(0) - points.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)
        
        # Radial basis
        K = dist ** 2 * torch.log(dist + 1e-8)
        
        # Affine part
        P = torch.ones(n, 3)
        P[:, 1] = points[:, 0]
        P[:, 2] = points[:, 1]
        
        # Build L matrix
        upper = torch.cat([K, P], dim=1)
        lower = torch.cat([P.T, torch.zeros(3, 3)], dim=1)
        L = torch.cat([upper, lower], dim=0)
        
        self.register_buffer('L_matrix', L)
        self.register_buffer('L_inverse', torch.inverse(L))
    
    def _precompute_radial_basis(self):
        """Pre-compute radial basis values for all pixels."""
        h, w = self.image_size
        device = self.control_points.device
        dtype = self.control_points.dtype
        
        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x_coords = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Flatten grid
        grid_flat = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
        self.register_buffer('grid_flat', grid_flat)
        
        # Compute distances to control points
        diff = grid_flat.unsqueeze(1) - self.control_points.unsqueeze(0)
        dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)
        
        # Radial basis
        U = dist ** 2 * torch.log(dist + 1e-8)
        self.register_buffer('radial_basis_values', U)
        
        # Affine part
        affine = torch.ones(h * w, 3, device=device, dtype=dtype)
        affine[:, 1] = grid_flat[:, 0]
        affine[:, 2] = grid_flat[:, 1]
        self.register_buffer('affine_part', affine)
        
        # Combined matrix
        combined = torch.cat([U, affine], dim=1)
        self.register_buffer('combined_matrix', combined)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply TPS warp (optimized)."""
        batch_size, channels, h, w = x.shape
        device = x.device
        
        # Get displaced points
        bounded_disp = torch.tanh(self.displacement) * self.max_displacement / h
        displaced = self.control_points + bounded_disp
        
        # Solve TPS using precomputed inverse
        rhs = torch.zeros(self.num_control_points + 3, 2, device=device)
        rhs[:self.num_control_points] = displaced
        
        coeffs = torch.matmul(self.L_inverse, rhs)  # (N+3, 2)
        
        # Create sampling grid using precomputed values
        transformed = torch.matmul(self.combined_matrix, coeffs)
        grid = transformed.view(h, w, 2).unsqueeze(0)
        
        # Apply transformation
        output = F.grid_sample(
            x, grid.expand(batch_size, -1, -1, -1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        return output
    
    def reset_parameters(self):
        """Reset displacement parameters."""
        if hasattr(self, 'displacement') and self.displacement.requires_grad:
            nn.init.zeros_(self.displacement)


class MultiScaleTPS(nn.Module):
    """
    Multi-scale TPS warp with hierarchical control points (Optimized).
    
    Applies TPS transformations at multiple scales for more
    flexible and detailed deformations. Each scale uses optimized
    precomputed matrices.
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        num_scales: Number of TPS scales to use.
        base_displacement: Base displacement magnitude.
        learnable: If True, all displacements are learnable.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        num_scales: int = 3,
        base_displacement: float = 15.0,
        learnable: bool = True
    ):
        super().__init__()
        self.num_scales = num_scales
        
        # Create TPS warps at different scales (all optimized)
        self.tps_scales = nn.ModuleList()
        for i in range(num_scales):
            grid_size = 2 + i  # 2x2, 3x3, 4x4, ...
            displacement = base_displacement / (i + 1)  # Smaller displacement at finer scales
            
            self.tps_scales.append(
                ThinPlateSplineGrid(
                    image_size=image_size,
                    grid_size=(grid_size, grid_size),
                    max_displacement=displacement,
                    learnable=learnable
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale TPS warp."""
        for tps in self.tps_scales:
            x = tps(x)
        return x
    
    def reset_parameters(self):
        """Reset all TPS parameters."""
        for tps in self.tps_scales:
            tps.reset_parameters()