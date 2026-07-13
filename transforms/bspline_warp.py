"""
B-Spline Warp Module.

This module implements a differentiable B-spline warp that deforms
the face region of an image while leaving the background untouched.

Key properties
--------------
* **Local support** — each control point only influences a 4×4
  neighbourhood of pixels (cubic B-spline), unlike TPS which has
  global influence.
* **Face mask blending** — a smooth mask derived from facial
  landmarks restricts the warp to the face area.  The background
  is copied from the original image unchanged.
* **Learnable** — the (dx, dy) displacement of every control point
  is a parameter θ that can be optimised with gradient descent.

The displacement field is:

    d(u, v) = Σᵢ Σⱼ Bᵢ(u) · Bⱼ(v) · δᵢⱼ

where B are cubic B-spline basis functions and δᵢⱼ are the
learnable control-point displacements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────
#  Cubic B-spline basis functions
# ──────────────────────────────────────────────────────────────

def _cubic_bspline_basis(t: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the four cubic B-spline basis functions at parameter t.

    The cubic B-spline has C² continuity and local support over
    the interval [0, 4) (i.e. each basis function is non-zero
    only on a span of 4 knot intervals).

    Parameters
    ----------
    t : (N,) tensor
        Parameter values in [0, 1) (normalised within one knot span).

    Returns
    -------
    (N, 4) tensor
        Values of B₋₁, B₀, B₁, B₂  (the four basis functions
        that are non-zero at t).
    """
    t2 = t * t
    t3 = t2 * t

    # Standard cubic B-spline basis (uniform knots)
    b0 = (1.0 - t) ** 3 / 6.0          # B₋₁
    b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0   # B₀
    b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0  # B₁
    b3 = t3 / 6.0                       # B₂

    return torch.stack([b0, b1, b2, b3], dim=-1)  # (N, 4)


# ──────────────────────────────────────────────────────────────
#  Face mask creation
# ──────────────────────────────────────────────────────────────

def create_face_mask(
    landmarks: torch.Tensor,
    image_size: Tuple[int, int],
    padding: float = 0.15,
    blur_sigma: float = 8.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Create a smooth face mask from facial landmarks.

    The mask is 1.0 inside the convex hull of the landmarks
    (padded) and fades smoothly to 0.0 outside, so that the
    warp blends seamlessly with the background.

    Parameters
    ----------
    landmarks : (N, 2) tensor
        Facial landmark coordinates in pixel space.
    image_size : (H, W)
        Image dimensions.
    padding : float
        Fractional padding around the landmark bounding box.
    blur_sigma : float
        Gaussian blur sigma for smooth edges.
    device : str
        Device for the output mask.

    Returns
    -------
    (1, 1, H, W) tensor
        Face mask in [0, 1].
    """
    h, w = image_size
    lm = landmarks.detach().cpu().numpy()

    # Bounding box of landmarks
    x_min, y_min = lm[:, 0].min(), lm[:, 1].min()
    x_max, y_max = lm[:, 0].max(), lm[:, 1].max()

    # Pad bounding box
    bw = x_max - x_min
    bh = y_max - y_min
    pad = max(bw, bh) * padding
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w, x_max + pad)
    y_max = min(h, y_max + pad)

    # Create binary mask from convex hull (or ellipse fallback)
    from scipy import ndimage
    mask = np.zeros((h, w), dtype=np.float32)

    # Use convex hull of landmarks for a tight face mask
    try:
        from scipy.spatial import ConvexHull
        if len(lm) >= 3:
            hull = ConvexHull(lm)
            hull_pts = lm[hull.vertices].astype(np.int32)
            # Fill the convex hull polygon
            from PIL import Image, ImageDraw
            pil_mask = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(pil_mask)
            hull_pts_list = [(int(p[0]), int(p[1])) for p in hull_pts]
            draw.polygon(hull_pts_list, fill=1)
            mask = np.array(pil_mask, dtype=np.float32)
    except Exception:
        # Fallback: ellipse from bounding box
        cy, cx = (y_min + y_max) / 2, (x_min + x_max) / 2
        ry, rx = (y_max - y_min) / 2, (x_max - x_min) / 2
        yy, xx = np.mgrid[0:h, 0:w]
        mask = ((xx - cx) ** 2 / (rx ** 2 + 1e-8) +
                (yy - cy) ** 2 / (ry ** 2 + 1e-8) <= 1).astype(np.float32)

    # Gaussian blur for smooth edges
    if blur_sigma > 0:
        mask = ndimage.gaussian_filter(mask, sigma=blur_sigma)

    # Normalise to [0, 1]
    if mask.max() > 0:
        mask = mask / mask.max()

    mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)
    return mask_t


# ──────────────────────────────────────────────────────────────
#  B-Spline Warp
# ──────────────────────────────────────────────────────────────

class BSplineWarp(nn.Module):
    """
    Differentiable B-spline warp with face-region masking.

    The image is parameterised by a grid of control points.
    Each control point has a learnable (dx, dy) displacement.
    The displacement at any pixel is a weighted sum of the
    four nearest control points' displacements, weighted by
    cubic B-spline basis functions.

    A face mask (computed from landmarks) restricts the warp
    to the face region — the background is copied unchanged.

    Args:
        image_size: (H, W) of the input images.
        grid_size: (rows, cols) of control points.
        max_displacement: Maximum pixel displacement per control point.
        learnable: If True, displacements are learnable parameters.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        grid_size: Tuple[int, int] = (8, 8),
        max_displacement: float = 10.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.grid_rows, self.grid_cols = grid_size
        self.num_control_points = self.grid_rows * self.grid_cols
        self.max_displacement = max_displacement
        self.learnable = learnable

        # Learnable displacements: (num_control_points, 2)  [dx, dy]
        if learnable:
            self.displacement = nn.Parameter(
                torch.zeros(self.num_control_points, 2),
                requires_grad=True,
            )
        else:
            self.register_buffer(
                'displacement',
                torch.zeros(self.num_control_points, 2),
            )

        # Pre-compute B-spline basis lookup tables
        self._precompute_basis()

        # Face mask (set externally via set_face_mask)
        self.register_buffer(
            'face_mask',
            torch.ones(1, 1, *image_size),
        )

    # ── Pre-computation ───────────────────────────────────────

    def _precompute_basis(self):
        """
        Pre-compute B-spline basis function values and control-point
        indices for every pixel.  This is done once at init time so
        the forward pass is a simple gather + matmul.
        """
        h, w = self.image_size
        gr, gc = self.grid_rows, self.grid_cols

        # Normalise pixel coordinates to [0, gc-1] x [0, gr-1]
        # so that integer values fall on control points
        xs = torch.linspace(0, gc - 1, w)
        ys = torch.linspace(0, gr - 1, h)

        # For each pixel, find the knot span and local parameter t
        # Control point index i = floor(x), t = x - i
        xi = torch.floor(xs).long()
        yi = torch.floor(ys).long()
        tx = xs - xi.float()
        ty = ys - yi.float()

        # Clamp indices to valid range
        xi = xi.clamp(0, gc - 1)
        yi = yi.clamp(0, gr - 1)

        # Evaluate basis functions: (W, 4) and (H, 4)
        bx = _cubic_bspline_basis(tx)  # (W, 4)
        by = _cubic_bspline_basis(ty)  # (H, 4)

        # For each pixel (py, px), the 4×4 = 16 control points that
        # influence it are at rows (yi[py]-1, yi[py], yi[py]+1, yi[py]+2)
        # and cols (xi[px]-1, xi[px], xi[px]+1, xi[px]+2).
        # We pre-compute the flattened indices and weights.

        # Row indices for each pixel row: (H, 4)
        row_offsets = torch.tensor([-1, 0, 1, 2])
        row_idx = yi.unsqueeze(1) + row_offsets.unsqueeze(0)  # (H, 4)
        row_idx = row_idx.clamp(0, gr - 1)

        # Col indices for each pixel col: (W, 4)
        col_offsets = torch.tensor([-1, 0, 1, 2])
        col_idx = xi.unsqueeze(1) + col_offsets.unsqueeze(0)  # (W, 4)
        col_idx = col_idx.clamp(0, gc - 1)

        # For each pixel (py, px), compute 16 control-point indices
        # and 16 weights (outer product of by and bx)
        # row_idx: (H, 4), col_idx: (W, 4)
        # by: (H, 4), bx: (W, 4)

        # Create full pixel grid: (H, W, 4, 4) -> flatten to (H*W, 16)
        yy, xx = torch.meshgrid(
            torch.arange(h), torch.arange(w), indexing='ij'
        )

        # Control point indices for each pixel: (H, W, 16)
        cp_rows = row_idx[yy]  # (H, W, 4)
        cp_cols = col_idx[xx]  # (H, W, 4)

        # Expand to (H, W, 4, 4) and flatten to (H*W, 16)
        cp_rows_exp = cp_rows.unsqueeze(3).expand(-1, -1, -1, 4)  # (H,W,4,4)
        cp_cols_exp = cp_cols.unsqueeze(2).expand(-1, -1, 4, -1)  # (H,W,4,4)

        # Flatten control point index: row * gc + col
        cp_flat = (cp_rows_exp * gc + cp_cols_exp).reshape(h * w, 16)

        # Weights: outer product of by[py] and bx[px]
        by_exp = by[yy]  # (H, W, 4)
        bx_exp = bx[xx]  # (H, W, 4)
        weights = (by_exp.unsqueeze(3) * bx_exp.unsqueeze(2))  # (H,W,4,4)
        weights = weights.reshape(h * w, 16)

        # Register as buffers
        self.register_buffer('cp_indices', cp_flat.long())
        self.register_buffer('cp_weights', weights.float())

    # ── Face mask ────────────────────────────────────────────

    def set_face_mask(self, mask: torch.Tensor):
        """
        Set the face mask that restricts the warp to the face region.

        Parameters
        ----------
        mask : (1, 1, H, W) tensor
            Face mask in [0, 1].
        """
        self.face_mask = mask.to(self.displacement.device)

    # ── Forward ───────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply B-spline warp to the face region of the input image.

        The background is copied unchanged from the input.

        Parameters
        ----------
        x : (B, C, H, W) tensor
            Input image.

        Returns
        -------
        (B, C, H, W) tensor
            Warped image (face only; background unchanged).
        """
        B, C, H, W = x.shape
        device = x.device

        # ── Compute displacement field ────────────────────────
        # Bounded displacement via tanh
        bounded_disp = torch.tanh(self.displacement) * self.max_displacement
        # bounded_disp: (num_cp, 2)

        # Gather displacements for each pixel: (H*W, 16, 2)
        pixel_disps = bounded_disp[self.cp_indices]  # (H*W, 16, 2)

        # Weighted sum: (H*W, 2)
        w = self.cp_weights.unsqueeze(-1)  # (H*W, 16, 1)
        displacement_field = (pixel_disps * w).sum(dim=1)  # (H*W, 2)

        # Reshape to (H, W, 2) and normalise to [-1, 1] for grid_sample
        disp_grid = displacement_field.view(H, W, 2)
        disp_grid = disp_grid / max(H, W)  # normalise to [-1, 1] range

        # Create sampling grid: identity + displacement
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        identity = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2)

        sampling_grid = identity + disp_grid  # (H, W, 2)
        sampling_grid = sampling_grid.clamp(-1.5, 1.5)  # allow slight overflow
        sampling_grid = sampling_grid.unsqueeze(0).expand(B, -1, -1, -1)

        # ── Warp the image ────────────────────────────────────
        warped = F.grid_sample(
            x, sampling_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )

        # ── Blend: face gets warped, background stays original ─
        mask = self.face_mask.to(device)  # (1, 1, H, W)
        output = mask * warped + (1.0 - mask) * x

        return output

    # ── Utilities ─────────────────────────────────────────────

    def reset_parameters(self):
        """Reset displacements to zero."""
        if self.learnable:
            nn.init.zeros_(self.displacement)

    def get_displacement_magnitude(self) -> float:
        """Get average displacement magnitude in pixels."""
        bounded = torch.tanh(self.displacement) * self.max_displacement
        return torch.norm(bounded, dim=1).mean().item()

    def extra_repr(self) -> str:
        return (
            f'image_size={self.image_size}, '
            f'grid_size=({self.grid_rows}, {self.grid_cols}), '
            f'num_control_points={self.num_control_points}, '
            f'max_displacement={self.max_displacement}, '
            f'learnable={self.learnable}'
        )
