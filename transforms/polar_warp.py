"""
Polar Coordinate Perturbation.

Perturbs the image in polar coordinate space: the radial coordinate r
and angular coordinate θ are displaced by learnable parameters.
This creates swirl / radial expansion / contraction effects.

The transform converts pixel (x, y) → polar (r, θ), applies learnable
displacements Δr(r, θ) and Δθ(r, θ), then converts back to Cartesian.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class PolarWarp(nn.Module):
    """
    Differentiable polar coordinate perturbation with face masking.

    Learnable parameters:
        radial_shift: (num_radial,) — radial displacement at each radial band
        angular_shift: (num_angular,) — angular displacement at each angular sector

    The displacement field is interpolated bilinearly from these parameters.

    Args:
        image_size: (H, W)
        num_radial: Number of radial control bands.
        num_angular: Number of angular control sectors.
        max_radial_shift: Max radial displacement (pixels).
        max_angular_shift: Max angular displacement (radians).
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        num_radial: int = 8,
        num_angular: int = 8,
        max_radial_shift: float = 10.0,
        max_angular_shift: float = 0.1,
        learnable: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.max_radial_shift = max_radial_shift
        self.max_angular_shift = max_angular_shift

        if learnable:
            self.radial_shift = nn.Parameter(
                torch.zeros(num_radial, num_angular), requires_grad=True
            )
            self.angular_shift = nn.Parameter(
                torch.zeros(num_radial, num_angular), requires_grad=True
            )
        else:
            self.register_buffer('radial_shift', torch.zeros(num_radial, num_angular))
            self.register_buffer('angular_shift', torch.zeros(num_radial, num_angular))

        self._precompute_polar_grid()
        self.register_buffer('face_mask', torch.ones(1, 1, *image_size))

    def _precompute_polar_grid(self):
        """Pre-compute polar coordinates and interpolation indices for all pixels."""
        h, w = self.image_size
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        max_r = np.sqrt(cx ** 2 + cy ** 2)

        y_coords = torch.arange(h).float()
        x_coords = torch.arange(w).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        yy = yy.contiguous()
        xx = xx.contiguous()

        # Cartesian → polar
        dx = xx - cx
        dy = yy - cy
        r = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        theta = torch.atan2(dy, dx)  # [-π, π]


        # Normalise r to [0, num_radial-1], theta to [0, num_angular-1]
        r_norm = (r / max_r) * (self.num_radial - 1)
        theta_norm = ((theta + np.pi) / (2 * np.pi)) * (self.num_angular - 1)

        self.register_buffer('r_norm', r_norm)
        self.register_buffer('theta_norm', theta_norm)
        self.register_buffer('r_pixel', r)
        self.register_buffer('theta', theta)
        self.register_buffer('max_r', torch.tensor(max_r))

        # Pre-compute interpolation indices (floor + fractional)
        r_idx = torch.floor(r_norm).long().clamp(0, self.num_radial - 1)
        t_idx = torch.floor(theta_norm).long().clamp(0, self.num_angular - 1)
        r_frac = (r_norm - r_idx.float())  # (H, W)
        t_frac = (theta_norm - t_idx.float())  # (H, W)


        # 4 neighbours for bilinear interpolation
        r_idx_p = (r_idx + 1).clamp(0, self.num_radial - 1)
        t_idx_p = (t_idx + 1).clamp(0, self.num_angular - 1)

        self.register_buffer('r_idx', r_idx)
        self.register_buffer('t_idx', t_idx)
        self.register_buffer('r_idx_p', r_idx_p)
        self.register_buffer('t_idx_p', t_idx_p)
        self.register_buffer('r_frac', r_frac)
        self.register_buffer('t_frac', t_frac)

    def set_face_mask(self, mask: torch.Tensor):
        self.face_mask = mask.to(self.radial_shift.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # Bounded shifts
        dr = torch.tanh(self.radial_shift) * self.max_radial_shift  # (nr, nt)
        dt = torch.tanh(self.angular_shift) * self.max_angular_shift  # (nr, nt)

        # Bilinear interpolation of shifts at each pixel
        def bilinear(field):
            v00 = field[self.r_idx, self.t_idx]
            v01 = field[self.r_idx, self.t_idx_p]
            v10 = field[self.r_idx_p, self.t_idx]
            v11 = field[self.r_idx_p, self.t_idx_p]
            return (v00 * (1 - self.r_frac) * (1 - self.t_frac) +
                    v01 * (1 - self.r_frac) * self.t_frac +
                    v10 * self.r_frac * (1 - self.t_frac) +
                    v11 * self.r_frac * self.t_frac)

        delta_r = bilinear(dr)  # (H, W)
        delta_t = bilinear(dt)  # (H, W)

        # Apply perturbation in polar space
        new_r = self.r_pixel + delta_r
        new_theta = self.theta + delta_t

        # Polar → Cartesian
        new_dx = new_r * torch.cos(new_theta)
        new_dy = new_r * torch.sin(new_theta)

        # Convert to normalised grid [-1, 1]
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        # ``grid_sample(..., align_corners=True)`` maps pixel coordinate 0 to
        # -1 and pixel coordinate (size - 1) to +1.  Dividing by ``size / 2``
        # shifts every nominally neutral sample and made an all-zero PolarWarp
        # visibly non-identity.  Use the exact align-corners normalization.
        grid_x = 2.0 * (new_dx + cx) / max(W - 1, 1) - 1.0
        grid_y = 2.0 * (new_dy + cy) / max(H - 1, 1) - 1.0

        sampling_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        sampling_grid = sampling_grid.unsqueeze(0).expand(B, -1, -1, -1)

        warped = F.grid_sample(
            x, sampling_grid, mode='bilinear',
            padding_mode='border', align_corners=True
        )

        mask = self.face_mask.to(device)
        return mask * warped + (1.0 - mask) * x

    def reset_parameters(self):
        nn.init.zeros_(self.radial_shift)
        nn.init.zeros_(self.angular_shift)

    def get_displacement_magnitude(self) -> float:
        dr = torch.tanh(self.radial_shift) * self.max_radial_shift
        return dr.abs().mean().item()
