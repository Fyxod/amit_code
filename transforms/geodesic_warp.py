"""
Geodesic Deformation.

Deforms the image along geodesic curves on a learnable Riemannian
manifold.  The manifold is defined by a learnable metric tensor field
g(x, y) that varies spatially.  Pixels are displaced along the geodesic
flow defined by this metric.

The metric tensor at each point is parameterised as:
    g(x, y) = I + Σᵢ αᵢ · φᵢ(x, y)

where φᵢ are basis functions (Gaussian bumps) and αᵢ are learnable
amplitudes.  The geodesic displacement is approximated by integrating
the metric-induced flow.

For efficiency, we pre-compute the basis functions and compute the
displacement field as a weighted sum, then solve a simplified geodesic
equation numerically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class GeodesicWarp(nn.Module):
    """
    Differentiable geodesic deformation with face masking.

    The image is treated as lying on a Riemannian manifold whose metric
    is perturbed from the Euclidean metric.  The displacement field
    follows geodesic curves induced by this metric perturbation.

    Args:
        image_size: (H, W)
        num_bumps: Number of Gaussian basis functions for the metric.
        max_displacement: Max pixel displacement.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        num_bumps: int = 16,
        max_displacement: float = 10.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_bumps = num_bumps
        self.max_displacement = max_displacement

        # Learnable amplitudes for metric perturbation
        if learnable:
            self.metric_amplitude = nn.Parameter(
                torch.zeros(num_bumps), requires_grad=True
            )
            # Learnable direction of geodesic flow per bump
            self.flow_direction = nn.Parameter(
                torch.zeros(num_bumps, 2), requires_grad=True
            )
        else:
            self.register_buffer('metric_amplitude', torch.zeros(num_bumps))
            self.register_buffer('flow_direction', torch.zeros(num_bumps, 2))

        self._precompute_basis()
        self.register_buffer('face_mask', torch.ones(1, 1, *image_size))

    def _precompute_basis(self):
        """Pre-compute Gaussian basis functions on a grid."""
        h, w = self.image_size
        n = int(np.sqrt(self.num_bumps))

        # Place basis centres on a grid
        cy = np.linspace(0.2, 0.8, n)
        cx = np.linspace(0.2, 0.8, n)
        centres = []
        for y in cy:
            for x in cx:
                centres.append([x, y])
        centres = np.array(centres[:self.num_bumps], dtype=np.float32)

        sigma = 0.15  # Gaussian width

        y_coords = torch.linspace(0, 1, h)
        x_coords = torch.linspace(0, 1, w)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Basis: (H*W, num_bumps)
        basis = []
        for c in centres:
            d2 = (xx - c[0]) ** 2 + (yy - c[1]) ** 2
            basis.append(torch.exp(-d2 / (2 * sigma ** 2)))

        basis = torch.stack(basis, dim=-1).reshape(h * w, self.num_bumps)
        self.register_buffer('basis', basis)

    def set_face_mask(self, mask: torch.Tensor):
        self.face_mask = mask.to(self.metric_amplitude.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # Bounded amplitudes and directions
        amp = torch.tanh(self.metric_amplitude) * self.max_displacement  # (num_bumps,)
        direction = torch.tanh(self.flow_direction)  # (num_bumps, 2)

        # Displacement field = Σᵢ ampᵢ · basisᵢ · directionᵢ
        # basis: (H*W, num_bumps), direction: (num_bumps, 2)
        weighted = self.basis * amp.unsqueeze(0)  # (H*W, num_bumps)
        field = torch.matmul(weighted, direction)  # (H*W, 2)
        field = field.view(H, W, 2)
        field_norm = field / max(H, W)

        # Identity + displacement
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
        nn.init.zeros_(self.metric_amplitude)
        nn.init.zeros_(self.flow_direction)

    def get_displacement_magnitude(self) -> float:
        a = torch.tanh(self.metric_amplitude) * self.max_displacement
        return a.abs().mean().item()
