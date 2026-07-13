"""
Laplacian Smoothing Perturbation.

Applies Laplacian smoothing to the displacement field.  The displacement
field is parameterised by a coarse grid of control points, and a Laplacian
smoothing operator is applied to ensure C² continuity.

The Laplacian operator Δ = ∂²/∂x² + ∂²/∂y² is discretised on the control
point grid.  The smoothing energy is:

    E_smooth = ½ ∫ |Δd|² dA

The displacement field is computed as:
    d = (I - λ·L)^(-1) · δ

where L is the discrete Laplacian matrix, δ are learnable control point
displacements, and λ controls the smoothing strength (also learnable).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class LaplacianSmoothingWarp(nn.Module):
    """
    Differentiable Laplacian-smoothed warp with face masking.

    The displacement field is defined on a coarse grid and smoothed
    via the Laplacian operator before being upsampled to full resolution.

    Args:
        image_size: (H, W)
        grid_size: (rows, cols) of control points.
        max_displacement: Max pixel displacement per control point.
        smoothing_lambda: Initial smoothing strength (learnable).
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        grid_size: Tuple[int, int] = (6, 6),
        max_displacement: float = 10.0,
        smoothing_lambda: float = 0.5,
        learnable: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.grid_rows, self.grid_cols = grid_size
        self.num_cp = self.grid_rows * self.grid_cols
        self.max_displacement = max_displacement

        if learnable:
            self.displacement = nn.Parameter(
                torch.zeros(self.num_cp, 2), requires_grad=True
            )
            self.smoothing_lambda = nn.Parameter(
                torch.tensor(smoothing_lambda), requires_grad=True
            )
        else:
            self.register_buffer('displacement', torch.zeros(self.num_cp, 2))
            self.register_buffer('smoothing_lambda', torch.tensor(smoothing_lambda))

        self._build_laplacian()
        self._precompute_upsample()
        self.register_buffer('face_mask', torch.ones(1, 1, *image_size))

    def _build_laplacian(self):
        """Build the discrete Laplacian matrix for the control point grid."""
        rows, cols = self.grid_rows, self.grid_cols
        n = rows * cols

        L = np.zeros((n, n), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                neighbours = []
                if i > 0:
                    neighbours.append((i - 1) * cols + j)
                if i < rows - 1:
                    neighbours.append((i + 1) * cols + j)
                if j > 0:
                    neighbours.append(i * cols + (j - 1))
                if j < cols - 1:
                    neighbours.append(i * cols + (j + 1))
                L[idx, idx] = len(neighbours)
                for nb in neighbours:
                    L[idx, nb] = -1.0

        self.register_buffer('L_matrix', torch.from_numpy(L))

    def _precompute_upsample(self):
        """Pre-compute bilinear upsampling from control grid to image."""
        h, w = self.image_size
        gr, gc = self.grid_rows, self.grid_cols

        # Normalise pixel coords to [0, gc-1] x [0, gr-1]
        xs = torch.linspace(0, gc - 1, w)
        ys = torch.linspace(0, gr - 1, h)

        xi = torch.floor(xs).long().clamp(0, gc - 1)
        yi = torch.floor(ys).long().clamp(0, gr - 1)
        tx = xs - xi.float()
        ty = ys - yi.float()

        xi_p = (xi + 1).clamp(0, gc - 1)
        yi_p = (yi + 1).clamp(0, gr - 1)

        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

        # 4 control point indices per pixel
        idx00 = (yi[yy] * gc + xi[xx]).reshape(h * w)
        idx01 = (yi[yy] * gc + xi_p[xx]).reshape(h * w)
        idx10 = (yi_p[yy] * gc + xi[xx]).reshape(h * w)
        idx11 = (yi_p[yy] * gc + xi_p[xx]).reshape(h * w)

        # Weights
        w00 = ((1 - tx[xx]) * (1 - ty[yy])).reshape(h * w, 1)
        w01 = (tx[xx] * (1 - ty[yy])).reshape(h * w, 1)
        w10 = ((1 - tx[xx]) * ty[yy]).reshape(h * w, 1)
        w11 = (tx[xx] * ty[yy]).reshape(h * w, 1)

        self.register_buffer('idx00', idx00)
        self.register_buffer('idx01', idx01)
        self.register_buffer('idx10', idx10)
        self.register_buffer('idx11', idx11)
        self.register_buffer('w00', w00)
        self.register_buffer('w01', w01)
        self.register_buffer('w10', w10)
        self.register_buffer('w11', w11)

    def set_face_mask(self, mask: torch.Tensor):
        self.face_mask = mask.to(self.displacement.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # Bounded displacement
        delta = torch.tanh(self.displacement) * self.max_displacement  # (n, 2)

        # Apply Laplacian smoothing: d = (I + λL)^(-1) · δ
        lam = torch.sigmoid(self.smoothing_lambda)  # [0, 1]
        I = torch.eye(self.num_cp, device=device, dtype=delta.dtype)
        smoothed_matrix = I + lam * self.L_matrix.to(device)
        smoothed_disp = torch.linalg.solve(smoothed_matrix, delta)

        # Bilinear upsample to full resolution
        v00 = smoothed_disp[self.idx00]
        v01 = smoothed_disp[self.idx01]
        v10 = smoothed_disp[self.idx10]
        v11 = smoothed_disp[self.idx11]

        field = (v00 * self.w00 + v01 * self.w01 +
                 v10 * self.w10 + v11 * self.w11)  # (H*W, 2)
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
        nn.init.zeros_(self.displacement)

    def get_displacement_magnitude(self) -> float:
        d = torch.tanh(self.displacement) * self.max_displacement
        return torch.norm(d, dim=1).mean().item()
