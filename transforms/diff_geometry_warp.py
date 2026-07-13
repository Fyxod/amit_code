"""
Differential Geometry Perturbation.

Perturbs the image using differential-geometric quantities computed from
a learnable scalar field f(x, y).  The displacement field is constructed
from the gradient ∇f, Hessian Hf, and Laplacian Δf of this field:

    d(x, y) = α · ∇f + β · (∇f)⊥ + γ · Δf · n̂

where:
    ∇f = (∂f/∂x, ∂f/∂y)     — gradient (direction of steepest ascent)
    (∇f)⊥ = (-∂f/∂y, ∂f/∂x)  — perpendicular gradient (level-set direction)
    Δf = ∂²f/∂x² + ∂²f/∂y²   — Laplacian (mean curvature)
    n̂ = unit normal

The scalar field f is parameterised as a sum of learnable Gaussian bumps:
    f(x, y) = Σᵢ wᵢ · exp(-||p - cᵢ||² / (2σ²))

The coefficients α, β, γ control the contribution of each differential
quantity and are also learnable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DiffGeometryWarp(nn.Module):
    """
    Differentiable differential-geometry warp with face masking.

    Computes gradient, perpendicular gradient, and Laplacian of a
    learnable scalar field to construct the displacement field.

    Args:
        image_size: (H, W)
        num_bumps: Number of Gaussian basis functions for the scalar field.
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

        if learnable:
            # Weights for the scalar field
            self.field_weights = nn.Parameter(
                torch.zeros(num_bumps), requires_grad=True
            )
            # Coefficients for gradient, perp-gradient, Laplacian contributions
            self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
            self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
            self.gamma = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        else:
            self.register_buffer('field_weights', torch.zeros(num_bumps))
            self.register_buffer('alpha', torch.tensor(0.5))
            self.register_buffer('beta', torch.tensor(0.5))
            self.register_buffer('gamma', torch.tensor(0.3))

        self._precompute_basis_and_derivatives()
        self.register_buffer('face_mask', torch.ones(1, 1, *image_size))

    def _precompute_basis_and_derivatives(self):
        """Pre-compute Gaussian basis functions and their derivatives."""
        h, w = self.image_size
        n = int(np.sqrt(self.num_bumps))

        cy = np.linspace(0.2, 0.8, n)
        cx = np.linspace(0.2, 0.8, n)
        centres = []
        for y in cy:
            for x in cx:
                centres.append([x, y])
        centres = np.array(centres[:self.num_bumps], dtype=np.float32)

        sigma = 0.15
        inv_2sigma2 = 1.0 / (2 * sigma ** 2)

        y_coords = torch.linspace(0, 1, h)
        x_coords = torch.linspace(0, 1, w)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Basis: φᵢ = exp(-d²/(2σ²))
        # ∂φ/∂x = -(x-cx)/σ² · φ
        # ∂φ/∂y = -(y-cy)/σ² · φ
        # ∂²φ/∂x² = ((x-cx)²/σ⁴ - 1/σ²) · φ
        # ∂²φ/∂y² = ((y-cy)²/σ⁴ - 1/σ²) · φ
        # Δφ = ∂²φ/∂x² + ∂²φ/∂y²

        basis_list = []
        dx_list = []
        dy_list = []
        lap_list = []

        for c in centres:
            dxc = xx - c[0]
            dyc = yy - c[1]
            d2 = dxc ** 2 + dyc ** 2
            phi = torch.exp(-d2 * inv_2sigma2)

            basis_list.append(phi)
            dx_list.append(-dxc * inv_2sigma2 * phi)
            dy_list.append(-dyc * inv_2sigma2 * phi)

            d2phi_dx2 = (dxc ** 2 * inv_2sigma2 ** 2 - inv_2sigma2) * phi
            d2phi_dy2 = (dyc ** 2 * inv_2sigma2 ** 2 - inv_2sigma2) * phi
            lap_list.append(d2phi_dx2 + d2phi_dy2)

        # Stack: each (H, W) → (H*W, num_bumps)
        basis = torch.stack(basis_list, dim=-1).reshape(h * w, self.num_bumps)
        dphi_dx = torch.stack(dx_list, dim=-1).reshape(h * w, self.num_bumps)
        dphi_dy = torch.stack(dy_list, dim=-1).reshape(h * w, self.num_bumps)
        lap_phi = torch.stack(lap_list, dim=-1).reshape(h * w, self.num_bumps)

        self.register_buffer('basis', basis)
        self.register_buffer('dphi_dx', dphi_dx)
        self.register_buffer('dphi_dy', dphi_dy)
        self.register_buffer('lap_phi', lap_phi)

    def set_face_mask(self, mask: torch.Tensor):
        self.face_mask = mask.to(self.field_weights.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # Bounded weights
        w = torch.tanh(self.field_weights)  # (num_bumps,)
        alpha = torch.tanh(self.alpha) * self.max_displacement
        beta = torch.tanh(self.beta) * self.max_displacement
        gamma = torch.tanh(self.gamma) * self.max_displacement

        # Compute differential quantities of the scalar field
        # f = Σ wᵢ φᵢ,  ∇f = (Σ wᵢ ∂φᵢ/∂x, Σ wᵢ ∂φᵢ/∂y)
        df_dx = torch.matmul(self.dphi_dx, w)  # (H*W,)
        df_dy = torch.matmul(self.dphi_dy, w)  # (H*W,)
        delta_f = torch.matmul(self.lap_phi, w)  # (H*W,)

        # Gradient direction
        grad_mag = torch.sqrt(df_dx ** 2 + df_dy ** 2 + 1e-8)
        nx = df_dx / grad_mag  # unit gradient x
        ny = df_dy / grad_mag  # unit gradient y

        # Perpendicular gradient (level-set direction)
        perp_x = -df_dy
        perp_y = df_dx

        # Displacement field
        disp_x = alpha * df_dx + beta * perp_x + gamma * delta_f * nx
        disp_y = alpha * df_dy + beta * perp_y + gamma * delta_f * ny

        field = torch.stack([disp_x, disp_y], dim=-1)  # (H*W, 2)
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
        nn.init.zeros_(self.field_weights)

    def get_displacement_magnitude(self) -> float:
        w = torch.tanh(self.field_weights)
        return w.abs().mean().item()
