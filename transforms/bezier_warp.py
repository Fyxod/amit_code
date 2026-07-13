"""
Bezier Curve Perturbation.

Represents the image displacement field as a set of Bezier curves.
The displacement of each pixel is determined by evaluating Bezier
basis functions at the pixel's normalised coordinates, weighted by
learnable control point displacements.

A Bezier curve of degree n is defined by n+1 control points P₀..Pₙ:
    B(t) = Σᵢ C(n,i) (1-t)^(n-i) t^i · Pᵢ

For 2D images we use a tensor-product Bezier surface:
    S(u, v) = Σᵢ Σⱼ Bᵢ(u) · Bⱼ(v) · δᵢⱼ

where B are Bernstein polynomials and δᵢⱼ are learnable (dx, dy).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def _bernstein_poly(n: int, i: int, t: torch.Tensor) -> torch.Tensor:
    """Bernstein polynomial basis: C(n,i) * t^i * (1-t)^(n-i)."""
    from math import comb
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


class BezierWarp(nn.Module):
    """
    Differentiable Bezier-surface warp with face masking.

    The displacement field is a tensor-product Bezier surface whose
    control points are learnable.  Each control point has a (dx, dy)
    displacement bounded by tanh.

    Args:
        image_size: (H, W)
        degree_u: Degree of Bezier in u (horizontal) direction.
        degree_v: Degree of Bezier in v (vertical) direction.
        max_displacement: Max pixel displacement per control point.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        degree_u: int = 5,
        degree_v: int = 5,
        max_displacement: float = 10.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.num_cp = (degree_u + 1) * (degree_v + 1)
        self.max_displacement = max_displacement

        if learnable:
            self.displacement = nn.Parameter(
                torch.zeros(self.num_cp, 2), requires_grad=True
            )
        else:
            self.register_buffer('displacement', torch.zeros(self.num_cp, 2))

        self._precompute_basis()
        self.register_buffer('face_mask', torch.ones(1, 1, *image_size))

    def _precompute_basis(self):
        """Pre-compute Bernstein basis values for every pixel."""
        h, w = self.image_size
        du, dv = self.degree_u, self.degree_v

        # Normalised coordinates [0, 1]
        u = torch.linspace(0, 1, w)
        v = torch.linspace(0, 1, h)

        # Bernstein basis: (W, du+1) and (H, dv+1)
        bu = torch.stack([_bernstein_poly(du, i, u) for i in range(du + 1)], dim=1)
        bv = torch.stack([_bernstein_poly(dv, i, v) for i in range(dv + 1)], dim=1)

        # Tensor product: (H, W, du+1, dv+1) → (H*W, num_cp)
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        bu_exp = bu[xx]  # (H, W, du+1)
        bv_exp = bv[yy]  # (H, W, dv+1)

        # Outer product → (H, W, du+1, dv+1)
        basis = bu_exp.unsqueeze(3) * bv_exp.unsqueeze(2)
        basis = basis.reshape(h * w, self.num_cp)

        self.register_buffer('basis', basis.float())

    def set_face_mask(self, mask: torch.Tensor):
        self.face_mask = mask.to(self.displacement.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # Bounded displacement
        disp = torch.tanh(self.displacement) * self.max_displacement  # (num_cp, 2)

        # Displacement field: (H*W, 2) = basis @ disp
        field = torch.matmul(self.basis, disp)  # (H*W, 2)
        field = field.view(H, W, 2)

        # Normalise to [-1, 1]
        field_norm = field / max(H, W)

        # Identity grid + displacement
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        identity = torch.stack([x_grid, y_grid], dim=-1)

        sampling_grid = identity + field_norm
        sampling_grid = sampling_grid.clamp(-1.5, 1.5)
        sampling_grid = sampling_grid.unsqueeze(0).expand(B, -1, -1, -1)

        warped = F.grid_sample(
            x, sampling_grid, mode='bilinear',
            padding_mode='border', align_corners=True
        )

        mask = self.face_mask.to(device)
        return mask * warped + (1.0 - mask) * x

    def reset_parameters(self):
        nn.init.zeros_(self.displacement)

    def get_displacement_magnitude(self) -> float:
        d = torch.tanh(self.displacement) * self.max_displacement
        return torch.norm(d, dim=1).mean().item()
