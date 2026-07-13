"""
Mobius Transform.

Applies a Mobius (fractional linear) transformation to the image plane.
A Mobius transform maps the complex plane via:

    w = (a*z + b) / (c*z + d)

where z = x + iy is the pixel coordinate (treated as a complex number)
and a, b, c, d are complex parameters with ad - bc ≠ 0.

Mobius transforms are conformal (angle-preserving) and map circles to
circles/lines.  They can create smooth global deformations that are
difficult for identity models to handle.

We parameterise a, b, c, d as learnable complex parameters (stored as
real + imaginary parts) with the identity transform as initial state:
a=1, b=0, c=0, d=1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class MobiusWarp(nn.Module):
    """
    Differentiable Mobius (fractional linear) transform with face masking.

    Args:
        image_size: (H, W)
        max_param: Max magnitude for each complex parameter component.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        max_param: float = 0.3,
        learnable: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.max_param = max_param

        # Complex parameters a, b, c, d stored as (real, imag) pairs
        # Identity: a=1+0i, b=0+0i, c=0+0i, d=1+0i
        if learnable:
            self.a = nn.Parameter(torch.tensor([1.0, 0.0]), requires_grad=True)
            self.b = nn.Parameter(torch.tensor([0.0, 0.0]), requires_grad=True)
            self.c = nn.Parameter(torch.tensor([0.0, 0.0]), requires_grad=True)
            self.d = nn.Parameter(torch.tensor([1.0, 0.0]), requires_grad=True)
        else:
            self.register_buffer('a', torch.tensor([1.0, 0.0]))
            self.register_buffer('b', torch.tensor([0.0, 0.0]))
            self.register_buffer('c', torch.tensor([0.0, 0.0]))
            self.register_buffer('d', torch.tensor([1.0, 0.0]))

        self._precompute_grid()
        self.register_buffer('face_mask', torch.ones(1, 1, *image_size))

    def _precompute_grid(self):
        """Pre-compute normalised coordinate grid as complex plane."""
        h, w = self.image_size
        y_coords = torch.linspace(-1, 1, h)
        x_coords = torch.linspace(-1, 1, w)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        y_grid = y_grid.contiguous()
        x_grid = x_grid.contiguous()

        self.register_buffer('x_grid', x_grid)
        self.register_buffer('y_grid', y_grid)


    @staticmethod
    def _complex_mul(a, b):
        """Multiply two complex numbers stored as (real, imag) tensors."""
        ar, ai = a[..., 0], a[..., 1]
        br, bi = b[..., 0], b[..., 1]
        return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)

    @staticmethod
    def _complex_div(a, b):
        """Divide two complex numbers stored as (real, imag) tensors."""
        ar, ai = a[..., 0], a[..., 1]
        br, bi = b[..., 0], b[..., 1]
        denom = br ** 2 + bi ** 2 + 1e-8
        return torch.stack([
            (ar * br + ai * bi) / denom,
            (ai * br - ar * bi) / denom
        ], dim=-1)

    def set_face_mask(self, mask: torch.Tensor):
        self.face_mask = mask.to(self.a.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # Bounded parameters (deviation from identity)
        a = torch.stack([
            1.0 + torch.tanh(self.a[0]) * self.max_param,
            torch.tanh(self.a[1]) * self.max_param
        ])
        b = torch.stack([
            torch.tanh(self.b[0]) * self.max_param,
            torch.tanh(self.b[1]) * self.max_param
        ])
        c = torch.stack([
            torch.tanh(self.c[0]) * self.max_param,
            torch.tanh(self.c[1]) * self.max_param
        ])
        d = torch.stack([
            1.0 + torch.tanh(self.d[0]) * self.max_param,
            torch.tanh(self.d[1]) * self.max_param
        ])

        # z = x + iy as (real, imag) tensor: (H, W, 2)
        z = torch.stack([self.x_grid, self.y_grid], dim=-1).to(device)

        # w = (a*z + b) / (c*z + d)
        az = self._complex_mul(a, z)  # (H, W, 2)
        numerator = az + b  # broadcast
        cz = self._complex_mul(c, z)
        denominator = cz + d
        w = self._complex_div(numerator, denominator)  # (H, W, 2)

        sampling_grid = w  # already in [-1, 1] range (approximately)
        sampling_grid = sampling_grid.clamp(-1.5, 1.5)
        sampling_grid = sampling_grid.unsqueeze(0).expand(B, -1, -1, -1)

        warped = F.grid_sample(
            x, sampling_grid, mode='bilinear',
            padding_mode='border', align_corners=True
        )

        mask = self.face_mask.to(device)
        return mask * warped + (1.0 - mask) * x

    def reset_parameters(self):
        nn.init.zeros_(self.a)
        self.a.data[0] = 1.0
        nn.init.zeros_(self.b)
        nn.init.zeros_(self.c)
        nn.init.zeros_(self.d)
        self.d.data[0] = 1.0

    def get_displacement_magnitude(self) -> float:
        return torch.tanh(self.b).abs().mean().item()
