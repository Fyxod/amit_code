"""
Delaunay Triangulation Warp Module (Optimized).

This module implements differentiable image warping using Delaunay triangulation.
Control points are displaced to create smooth geometric distortions within each triangle.

OPTIMIZED VERSION:
- Pre-computes triangulation mesh once during initialization
- Caches barycentric coordinates for all pixels
- Only updates vertex positions during forward pass (no re-triangulation)
- Vectorized grid computation instead of Python loops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import Delaunay
from typing import Optional, Tuple, List


class DelaunayWarp(nn.Module):
    """
    Optimized Differentiable Delaunay triangulation warp.
    
    Creates a mesh of triangles from control points once during initialization
    and only updates vertex positions during optimization, avoiding expensive
    re-triangulation in each iteration.
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        num_points: Number of control points (including corners).
        max_displacement: Maximum pixel displacement for control points.
        learnable: If True, displacements are learnable parameters.
        border_mode: Border handling mode ('constant', 'reflect', 'replicate').
        include_corners: If True, include image corners as fixed control points.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        num_points: int = 16,
        max_displacement: float = 10.0,
        learnable: bool = True,
        border_mode: str = 'constant',
        include_corners: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.num_points = num_points
        self.max_displacement = max_displacement
        self.learnable = learnable
        self.border_mode = border_mode
        self.include_corners = include_corners
        
        # Initialize control points and triangulation (done once)
        self._init_control_points()
        self._compute_triangulation()
        self._init_displacement_parameters()
        
        # Pre-compute all interpolation data (done once)
        self._precompute_interpolation_data()
    
    def _init_control_points(self):
        """Initialize control points on a grid."""
        h, w = self.image_size
        
        # Calculate grid dimensions
        n = int(np.sqrt(self.num_points))
        
        # Create grid of control points
        y_coords = np.linspace(0, h - 1, n)
        x_coords = np.linspace(0, w - 1, n)
        
        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([x, y])
        
        # Add corner points if needed
        if self.include_corners:
            corners = [
                [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]
            ]
            for corner in corners:
                if corner not in points:
                    points.append(corner)
        
        self.base_points = np.array(points, dtype=np.float32)
        self.num_control_points = len(self.base_points)
        
        # Register as buffer
        self.register_buffer(
            'base_points_tensor',
            torch.from_numpy(self.base_points)
        )
    
    def _compute_triangulation(self):
        """Compute Delaunay triangulation of control points (done once)."""
        # Use scipy for Delaunay triangulation - only done once!
        self.triangulation = Delaunay(self.base_points)
        self.triangles = self.triangulation.simplices
        
        # Register triangles as buffer (static)
        self.register_buffer(
            'triangles_tensor',
            torch.from_numpy(self.triangles).long()
        )
    
    def _init_displacement_parameters(self):
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
    
    def _precompute_interpolation_data(self):
        """
        Pre-compute all interpolation data once during initialization.
        
        This is the key optimization - we compute:
        1. Which triangle each pixel belongs to
        2. Barycentric coordinates for each pixel
        3. Vertex indices for vectorized computation
        """
        h, w = self.image_size
        num_pixels = h * w
        
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing='ij'
        )
        
        # Flatten coordinates
        coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)
        
        # Find which triangle each pixel belongs to (using scipy)
        simplex_indices = self.triangulation.find_simplex(coords.numpy())
        
        # Register simplex indices (static)
        self.register_buffer(
            'simplex_indices',
            torch.from_numpy(simplex_indices).long()
        )
        
        # Pre-compute barycentric coordinates and vertex indices
        bary_coords = np.zeros((num_pixels, 3), dtype=np.float32)
        vertex_indices = np.zeros((num_pixels, 3), dtype=np.int64)
        
        for i in range(num_pixels):
            simplex_idx = simplex_indices[i]
            
            if simplex_idx >= 0:
                # Get triangle vertices
                triangle = self.triangles[simplex_idx]
                vertices = self.base_points[triangle]
                
                # Store vertex indices
                vertex_indices[i] = triangle
                
                # Compute barycentric coordinates
                x, y = coords[i, 0].item(), coords[i, 1].item()
                bary_coords[i] = self._point_to_barycentric(x, y, vertices)
        
        # Register precomputed data (static)
        self.register_buffer(
            'barycentric_coords',
            torch.from_numpy(bary_coords)
        )
        self.register_buffer(
            'vertex_indices',
            torch.from_numpy(vertex_indices).long()
        )
        
        # Pre-compute pixel coordinates for output
        pixel_y, pixel_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing='ij'
        )
        self.register_buffer('pixel_x', pixel_x.flatten())
        self.register_buffer('pixel_y', pixel_y.flatten())
    
    def _point_to_barycentric(
        self, 
        x: float, 
        y: float, 
        vertices: np.ndarray
    ) -> np.ndarray:
        """
        Convert point to barycentric coordinates within a triangle.
        
        Args:
            x, y: Point coordinates
            vertices: Triangle vertices (3, 2)
            
        Returns:
            Barycentric coordinates (3,)
        """
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]
        
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        
        if abs(denom) < 1e-10:
            return np.array([1/3, 1/3, 1/3])
        
        lambda1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
        lambda2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
        lambda3 = 1 - lambda1 - lambda2
        
        return np.array([lambda1, lambda2, lambda3])
    
    def get_displaced_points(self) -> torch.Tensor:
        """
        Get control points with applied displacements.
        
        Returns:
            Displaced control points tensor of shape (N, 2)
        """
        # Apply tanh to bound displacement
        bounded_displacement = torch.tanh(self.displacement) * self.max_displacement
        
        # Add displacement to base points
        displaced = self.base_points_tensor + bounded_displacement
        
        return displaced
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Delaunay warp to input images (optimized - no re-triangulation).
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Warped tensor of shape (B, C, H, W)
        """
        batch_size, channels, h, w = x.shape
        device = x.device
        
        # Get displaced control points
        displaced_points = self.get_displaced_points()
        
        # Vectorized computation of sampling grid
        grid = self._compute_sampling_grid_vectorized(displaced_points, h, w, device)
        
        # Apply grid sample
        output = F.grid_sample(
            x, grid.expand(batch_size, -1, -1, -1),
            mode='bilinear',
            padding_mode='zeros' if self.border_mode == 'constant' else self.border_mode,
            align_corners=True
        )
        
        return output
    
    def _compute_sampling_grid_vectorized(
        self, 
        displaced_points: torch.Tensor,
        h: int, 
        w: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute sampling grid using vectorized operations (no Python loops).
        
        This is the key optimization - uses precomputed barycentric coordinates
        and vertex indices for fast computation.
        
        Args:
            displaced_points: Displaced control points (N, 2)
            h, w: Image dimensions
            device: Device
            
        Returns:
            Sampling grid of shape (1, H, W, 2)
        """
        num_pixels = h * w
        
        # Get displaced vertex positions for each pixel
        # vertex_indices: (num_pixels, 3) - indices of triangle vertices
        # displaced_points: (num_control_points, 2) - displaced positions
        
        # Gather displaced vertex positions for each pixel
        # Shape: (num_pixels, 3, 2)
        displaced_vertices = displaced_points[self.vertex_indices]
        
        # Compute source positions using barycentric interpolation
        # barycentric_coords: (num_pixels, 3)
        # displaced_vertices: (num_pixels, 3, 2)
        
        # Expand barycentric coords for broadcasting: (num_pixels, 3, 1)
        bary_expanded = self.barycentric_coords.unsqueeze(-1)
        
        # Compute weighted sum: (num_pixels, 2)
        src_positions = (displaced_vertices * bary_expanded).sum(dim=1)
        
        # Handle pixels outside triangulation (simplex_idx < 0)
        # Use identity mapping for these pixels
        outside_mask = (self.simplex_indices < 0).unsqueeze(-1)
        identity_positions = torch.stack([self.pixel_x, self.pixel_y], dim=1).float()
        
        src_positions = torch.where(outside_mask, identity_positions.to(device), src_positions)
        
        # Normalize to [-1, 1] for grid_sample
        # Reshape src_positions to (H, W, 2) first
        grid = src_positions.view(h, w, 2)
        
        # Normalize x and y coordinates
        grid[..., 0] = 2.0 * grid[..., 0] / (w - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (h - 1) - 1.0
        
        return grid.unsqueeze(0)
    
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
        return (f'image_size={self.image_size}, num_points={self.num_control_points}, '
                f'max_displacement={self.max_displacement}, learnable={self.learnable}')


class AdaptiveDelaunayWarp(nn.Module):
    """
    Adaptive Delaunay warp with content-aware control point placement.
    
    Places more control points in regions with higher detail/importance.
    
    Args:
        image_size: Tuple of (height, width) for the input images.
        num_points: Number of control points.
        max_displacement: Maximum pixel displacement.
        learnable: If True, displacements are learnable.
        adaptive: If True, adapt control points to image content.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        num_points: int = 16,
        max_displacement: float = 10.0,
        learnable: bool = True,
        adaptive: bool = True
    ):
        super().__init__()
        self.adaptive = adaptive
        
        # Base Delaunay warp (optimized)
        self.base_warp = DelaunayWarp(
            image_size=image_size,
            num_points=num_points,
            max_displacement=max_displacement,
            learnable=learnable
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive Delaunay warp."""
        if self.adaptive:
            # Compute importance map (gradient magnitude)
            importance = self._compute_importance(x)
            # Could use importance to adapt control points
            # For now, use base warp
        return self.base_warp(x)
    
    def _compute_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute importance map based on image content."""
        # Sobel gradients
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=x.device
        ).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=x.device
        ).view(1, 1, 3, 3)
        
        # Convert to grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Compute gradients
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        
        # Gradient magnitude
        importance = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        return importance