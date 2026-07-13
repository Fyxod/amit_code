"""
Lens Barrel-Pincushion Distortion.

Models radial lens distortion using the Brown-Conrady model:
    r_d = r * (1 + k1*r² + k2*r⁴ + k3*r⁶)

Positive k → barrel distortion (pincushion when negative).
Also includes tangential distortion terms for decentering:
    x_d = x + (2*p1*x*y + p2*(r² + 2*x²))
    y_d = y + (p1*(r² + 2*y²) + 2*p2*x*y)

All coefficients k1, k2, k3, p1, p2 are learnable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class LensDistortion(nn.Module):
    """
    Differentiable lens barrel-pincushion distortion with face masking.

    Args:
        image_size: (H, W)
        max_k: Max magnitude for radial distortion coefficients.
        max_p: Max magnitude for tangential distortion coefficients.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        max_k: float = 0.5,
        max_p: float = 0.1,
        learnable: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.max_k = max_k
        self.max_p = max_p

        if learnable:
            self.k1 = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.k2 = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.k3 = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.p1 = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.p2 = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            for name in ['k1', 'k2', 'k3', 'p1', 'p2']:
                self.register_buffer(name, torch.zeros(1))

        self._precompute_grid()
        self.register_buffer('face_mask', torch.ones(1, 1, *image_size))

    def _precompute_grid(self):
        """Pre-compute normalised coordinate grid."""
        h, w = self.image_size
        y_coords = torch.linspace(-1, 1, h)
        x_coords = torch.linspace(-1, 1, w)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        y_grid = y_grid.contiguous()
        x_grid = x_grid.contiguous()

        r2 = x_grid ** 2 + y_grid ** 2
        r = torch.sqrt(r2 + 1e-8)

        self.register_buffer('x_grid', x_grid)
        self.register_buffer('y_grid', y_grid)
        self.register_buffer('r2', r2)
        self.register_buffer('r', r)


    def set_face_mask(self, mask: torch.Tensor):
        self.face_mask = mask.to(self.k1.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # Bounded coefficients
        k1 = torch.tanh(self.k1) * self.max_k
        k2 = torch.tanh(self.k2) * self.max_k
        k3 = torch.tanh(self.k3) * self.max_k
        p1 = torch.tanh(self.p1) * self.max_p
        p2 = torch.tanh(self.p2) * self.max_p

        xg = self.x_grid
        yg = self.y_grid
        r2 = self.r2

        # Radial distortion factor
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3

        # Tangential distortion
        tx = 2 * p1 * xg * yg + p2 * (r2 + 2 * xg ** 2)
        ty = p1 * (r2 + 2 * yg ** 2) + 2 * p2 * xg * yg

        x_d = xg * radial + tx
        y_d = yg * radial + ty

        sampling_grid = torch.stack([x_d, y_d], dim=-1)
        sampling_grid = sampling_grid.clamp(-1.5, 1.5)
        sampling_grid = sampling_grid.unsqueeze(0).expand(B, -1, -1, -1)

        warped = F.grid_sample(
            x, sampling_grid, mode='bilinear',
            padding_mode='border', align_corners=True
        )

        mask = self.face_mask.to(device)
        return mask * warped + (1.0 - mask) * x

    def reset_parameters(self):
        for p in [self.k1, self.k2, self.k3, self.p1, self.p2]:
            nn.init.zeros_(p)

    def get_displacement_magnitude(self) -> float:
        k1 = torch.tanh(self.k1) * self.max_k
        return k1.abs().item()
