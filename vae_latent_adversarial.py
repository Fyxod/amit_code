#!/usr/bin/env python
"""
VAE Latent Optimization for Adversarial Geometric Perturbations.

This script uses the InstructPix2Pix VAE (or TAESD fallback) to optimise a
perturbation in the *latent* space rather than directly on warp parameters.
A learnable latent delta Δz is added to the encoded original image, decoded
back to pixel space, and then a fixed geometric warp is applied.  The
objective is to disrupt identity embeddings while preserving facial
landmarks.

Pipeline:
    1. Load image → encode to z_orig via VAE encoder (frozen).
    2. Initialise learnable Δz = 0 (same shape as z_orig).
    3. Warp parameters are learnable (requires_grad=True), bounded by
       per-warp-type imperceptibility thresholds via internal tanh().
    4. For each iteration:
       a. z_perturbed = z_orig + Δz
       b. perturbed = VAE.decode(z_perturbed)
       c. warped = warp(perturbed, θ_warp)  # θ_warp is also optimised
       d. E_pert = extract_embedding(warped)
       e. P_pert = detect_landmarks(warped)       # non-diff, monitor only
       f. L_ssim  = 1 - SSIM(perturbed, orig)     # differentiable preservation
       g. L_pixel = ||perturbed - orig||²          # differentiable preservation
       h. L_total = L_identity + λ_ssim·L_ssim + λ_pix·L_pixel
                   + λ_lm·L_landmark + λ_reg·||Δz||²
       i. Backprop → update Δz AND θ_warp (jointly)
    5. Save perturbed + warped images, loss curves, visualisations.


Usage:
    python vae_latent_adversarial.py -i original.jpg --warp-type bspline
    python vae_latent_adversarial.py -i original.jpg --warp-type all
    python vae_latent_adversarial.py -i original.jpg --warp-type tps --visualize
"""

import argparse
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ── Local imports ──────────────────────────────────────────────
from transforms import (
    BSplineWarp,
    PolarWarp,
    BezierWarp,
    LensDistortion,
    MobiusWarp,
    LaplacianSmoothingWarp,
    GeodesicWarp,
    DiffGeometryWarp,
    DelaunayWarp,
    ThinPlateSpline,
    RollingShutter,
    FFTPhasePerturbation,
    HomographyTransform,
    create_face_mask,
)
from models import FaceRecognitionModel
from utils import load_image, save_image, Visualizer
from utils.landmarks import LandmarkDetector


# ──────────────────────────────────────────────────────────────
#  Warp registry  (same as bspline_adversarial.py)
# ──────────────────────────────────────────────────────────────

WARP_TYPES = {
    "bspline":   BSplineWarp,
    "polar":     PolarWarp,
    "bezier":    BezierWarp,
    "lens":      LensDistortion,
    "mobius":    MobiusWarp,
    "laplacian": LaplacianSmoothingWarp,
    "geodesic":  GeodesicWarp,
    "diffgeom":  DiffGeometryWarp,
    "delaunay":  DelaunayWarp,
    "tps":       ThinPlateSpline,
    "rolling":   RollingShutter,
    "fft":       FFTPhasePerturbation,
    "homography": HomographyTransform,
}

# ──────────────────────────────────────────────────────────────
#  Per-warp-type imperceptibility thresholds
#
#  These set the `max_*` attributes on each warp module.  Since every
#  warp uses `tanh(param) * max_*` internally, these act as hard
#  bounds on the actual geometric displacement — guaranteeing visual
#  imperceptibility regardless of what the optimizer does.
#
#  The `imperceptibility_scale` CLI arg multiplies all values, so
#  1.0 = conservative (default), 2.0 = looser, 0.5 = tighter.
# ──────────────────────────────────────────────────────────────

IMPERCEPTIBILITY_THRESHOLDS = {
    "bspline":    {"max_displacement": 3.0},
    "polar":      {"max_radial_shift": 3.0, "max_angular_shift": 0.03},
    "bezier":     {"max_displacement": 3.0},
    "lens":       {"max_k": 0.5, "max_p": 0.05},
    "mobius":     {"max_param": 0.15},
    "laplacian":  {"max_displacement": 3.0, "smoothing_lambda": 0.5},
    "geodesic":   {"max_displacement": 3.0},
    "diffgeom":   {"max_displacement": 3.0},
    "delaunay":   {"max_displacement": 3.0},
    "tps":        {"max_displacement": 3.0},
    "rolling":    {"max_offset": 2.0},
    "fft":        {"magnitude": 0.15},
    "homography": {"max_perturbation": 0.03},
}


def create_warp(warp_type, image_size, grid_size, imperceptibility_scale=1.0):
    """
    Create a warp module with learnable parameters bounded by
    per-warp-type imperceptibility thresholds.

    The warp's internal `tanh(param) * max_*` ensures the actual
    geometric displacement never exceeds the threshold, keeping
    perturbations visually imperceptible.

    Args:
        warp_type: One of WARP_TYPES keys.
        image_size: (H, W) tuple.
        grid_size: (rows, cols) for control-point-based warps.
        imperceptibility_scale: Multiplier on threshold values
            (1.0 = conservative default).

    Returns:
        nn.Module with requires_grad=True parameters.
    """
    thresholds = IMPERCEPTIBILITY_THRESHOLDS[warp_type].copy()
    # Scale all numeric thresholds
    for k, v in thresholds.items():
        thresholds[k] = v * imperceptibility_scale

    if warp_type == "bspline":
        w = BSplineWarp(image_size=image_size, grid_size=grid_size,
                        max_displacement=thresholds["max_displacement"], learnable=True)
    elif warp_type == "polar":
        w = PolarWarp(image_size=image_size, num_radial=grid_size[0],
                      num_angular=grid_size[1],
                      max_radial_shift=thresholds["max_radial_shift"],
                      max_angular_shift=thresholds["max_angular_shift"],
                      learnable=True)
    elif warp_type == "bezier":
        w = BezierWarp(image_size=image_size, degree_u=grid_size[0] - 1,
                       degree_v=grid_size[1] - 1,
                       max_displacement=thresholds["max_displacement"], learnable=True)
    elif warp_type == "lens":
        w = LensDistortion(image_size=image_size,
                           max_k=thresholds["max_k"], max_p=thresholds["max_p"],
                           learnable=True)
    elif warp_type == "mobius":
        w = MobiusWarp(image_size=image_size,
                       max_param=thresholds["max_param"], learnable=True)
    elif warp_type == "laplacian":
        w = LaplacianSmoothingWarp(image_size=image_size, grid_size=grid_size,
                                   max_displacement=thresholds["max_displacement"],
                                   smoothing_lambda=thresholds.get("smoothing_lambda", 0.5),
                                   learnable=True)
    elif warp_type == "geodesic":
        w = GeodesicWarp(image_size=image_size,
                         num_bumps=grid_size[0] * grid_size[1],
                         max_displacement=thresholds["max_displacement"],
                         learnable=True)
    elif warp_type == "diffgeom":
        w = DiffGeometryWarp(image_size=image_size,
                             num_bumps=grid_size[0] * grid_size[1],
                             max_displacement=thresholds["max_displacement"],
                             learnable=True)
    elif warp_type == "delaunay":
        w = DelaunayWarp(image_size=image_size,
                         num_points=grid_size[0] * grid_size[1],
                         max_displacement=thresholds["max_displacement"],
                         learnable=True)
    elif warp_type == "tps":
        w = ThinPlateSpline(image_size=image_size,
                            num_control_points=grid_size[0] * grid_size[1],
                            max_displacement=thresholds["max_displacement"],
                            learnable=True)
    elif warp_type == "rolling":
        w = RollingShutter(image_size=image_size,
                           max_offset=thresholds["max_offset"],
                           direction='horizontal', wave_type='sine',
                           num_harmonics=4, learnable=True)
    elif warp_type == "fft":
        w = FFTPhasePerturbation(image_size=image_size,
                                 magnitude=thresholds["magnitude"],
                                 phase_resolution=grid_size, shared_channels=True,
                                 learnable=True)
    elif warp_type == "homography":
        w = HomographyTransform(image_size=image_size,
                                max_perturbation=thresholds["max_perturbation"],
                                learnable=True)
    else:
        raise ValueError(f"Unknown warp type: {warp_type}")

    # Ensure all parameters are learnable
    for p in w.parameters():
        p.requires_grad = True

    return w


def create_all_warps(image_size, grid_size, device, imperceptibility_scale=1.0):
    """Create one learnable warp module per type and move to device."""
    warps = {}
    for wt in WARP_TYPES:
        w = create_warp(wt, image_size, grid_size, imperceptibility_scale).to(device)
        warps[wt] = w
    return warps


def get_warp_magnitude(warp, warp_type):
    """Get the current perturbation magnitude of a warp module."""
    if hasattr(warp, 'get_displacement_magnitude'):
        return warp.get_displacement_magnitude()
    elif hasattr(warp, 'get_perturbation_magnitude'):
        return warp.get_perturbation_magnitude()
    else:
        return 0.0



# ──────────────────────────────────────────────────────────────
#  VAE wrapper (InstructPix2Pix VAE with TAESD fallback)
# ──────────────────────────────────────────────────────────────

class VAELatentOptimiser(nn.Module):
    """
    VAE latent-space optimiser.

    Encodes the original image to a latent z_orig (frozen), creates a
    learnable delta Δz, and decodes z_orig + Δz back to pixel space.

    The VAE can be:
      * InstructPix2Pix VAE (via diffusers AutoencoderKL)
      * TAESD (via diffusers AutoencoderTiny) — local fallback

    Args:
        model_id: HuggingFace model ID for InstructPix2Pix.
        taesd_path: Local path to TAESD directory (fallback).
        device: Device to run on.
        scaling_factor: Latent scaling factor (1.0 for TAESD, ~0.18215 for SD VAE).
    """

    def __init__(
        self,
        model_id: str = "timbrooks/instruct-pix2pix",
        taesd_path: str = "taesd",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.vae_type = None        # 'pix2pix' or 'taesd'
        self.scaling_factor = 1.0   # will be set per VAE type
        self.vae = self._load_vae(model_id, taesd_path)

    # ── Loading ───────────────────────────────────────────────

    def _load_vae(self, model_id: str, taesd_path: str) -> nn.Module:
        """Try InstructPix2Pix VAE, fall back to TAESD."""
        # Attempt 1: InstructPix2Pix VAE via diffusers
        try:
            from diffusers import AutoencoderKL
            print(f"Loading InstructPix2Pix VAE from '{model_id}' ...")
            vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
            self.vae_type = "pix2pix"
            self.scaling_factor = 0.18215  # standard SD VAE scaling
            print("  ✓ InstructPix2Pix VAE loaded.")
            return vae.to(self.device)
        except Exception as e:
            print(f"  Could not load InstructPix2Pix VAE: {e}")

        # Attempt 2: TAESD (local)
        try:
            from diffusers import AutoencoderTiny
            taesd_full = os.path.join(taesd_path)
            if os.path.exists(taesd_full):
                print(f"Falling back to TAESD from '{taesd_path}' ...")
                vae = AutoencoderTiny.from_pretrained(taesd_path)
                self.vae_type = "taesd"
                self.scaling_factor = 1.0
                print("  ✓ TAESD VAE loaded.")
                return vae.to(self.device)
        except Exception as e:
            print(f"  Could not load TAESD: {e}")

        # Attempt 3: Simple CNN autoencoder fallback
        print("  Falling back to simple CNN autoencoder.")
        vae = SimpleVAE().to(self.device)
        self.vae_type = "simple"
        self.scaling_factor = 1.0
        return vae

    # ── Encode / Decode ───────────────────────────────────────

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image (B,3,H,W) in [0,1] → latent z (B,C,H',W')."""
        # Normalise to [-1, 1] for VAE
        x = 2.0 * image - 1.0

        if self.vae_type == "pix2pix":
            z = self.vae.encode(x).latent_dist.sample()
            z = z * self.scaling_factor
        elif self.vae_type == "taesd":
            # AutoencoderTiny.encode() returns AutoencoderTinyOutput
            out = self.vae.encode(x)
            z = out.latents if hasattr(out, 'latents') else out
        else:
            z = self.vae.encode(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z (B,C,H',W') → image (B,3,H,W) in [0,1]."""
        if self.vae_type == "pix2pix":
            x = self.vae.decode(z / self.scaling_factor).sample
        elif self.vae_type == "taesd":
            out = self.vae.decode(z)
            x = out.sample if hasattr(out, 'sample') else out
        else:
            x = self.vae.decode(z)
        # Convert from [-1, 1] to [0, 1]
        x = (x / 2 + 0.5).clamp(0, 1)
        return x


class SimpleVAE(nn.Module):
    """Minimal CNN autoencoder fallback when neither pix2pix nor TAESD loads."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, 3, stride=1, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 256, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# ──────────────────────────────────────────────────────────────
#  Embedding extraction & loss functions
#  (same logic as bspline_adversarial.py but gradient flows to Δz)
# ──────────────────────────────────────────────────────────────

def extract_embedding(identity_model, x):
    """Extract identity embedding with gradient flow."""
    if x.shape[2:] != identity_model.input_size:
        x = F.interpolate(x, size=identity_model.input_size,
                          mode='bilinear', align_corners=False)
    embedding = identity_model.model(x)
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding


def identity_loss(E_orig, E_pert):
    """L_identity = cos(E_orig, E_pert) — minimise to disrupt identity."""
    return F.cosine_similarity(E_orig, E_pert, dim=1).mean()


def landmark_loss(P_orig, P_pert):
    """L_landmark = mean ||P_pert - P_orig||₂."""
    displacement = P_pert - P_orig
    magnitude = torch.norm(displacement, dim=2)
    return magnitude.mean()


def latent_regularisation(dz):
    """L_reg = ||Δz||² — keep latent perturbation small."""
    return torch.mean(dz ** 2)


def pixel_l2_loss(perturbed, original):
    """L_pixel = mean ||perturbed - original||² — pixel-level preservation.

    Differentiable: gradient flows through perturbed → VAE.decode → Δz.
    """
    return torch.mean((perturbed - original) ** 2)


def _gaussian_kernel(window_size=11, sigma=1.5, channels=3, device='cpu'):
    """Create a 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    kernel = kernel.expand(channels, 1, window_size, window_size).contiguous()
    return kernel


def ssim_loss(x, y, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Differentiable SSIM loss (1 - SSIM).

    Acts as a differentiable proxy for structural/landmark preservation.
    Gradient flows through both x and y.

    Args:
        x, y: (B, C, H, W) images in [0, 1].
    Returns:
        Scalar loss in [0, 1] (0 = identical, 1 = completely different).
    """
    channels = x.shape[1]
    device = x.device
    kernel = _gaussian_kernel(window_size, channels=channels, device=device)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=channels)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=channels) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return 1.0 - ssim_map.mean()



# ──────────────────────────────────────────────────────────────
#  Main optimisation loop
# ──────────────────────────────────────────────────────────────

def run_vae_latent_optimisation(
    image,
    vae_optimiser,
    warps,
    identity_model,
    landmark_detector,
    *,
    warp_type="bspline",
    num_iterations=500,
    learning_rate=0.01,
    landmark_weight=0.1,
    latent_reg_weight=0.001,
    ssim_weight=5.0,
    pixel_weight=1.0,
    epsilon=10.0,
    device="cuda",
    save_dir="results/vae_latent_iterations",
    save_interval=50,
    verbose=True,
    whole_image=False,
):
    """
    Run joint VAE latent + warp parameter optimisation.

    Both Δz (latent delta) and warp parameters θ_warp are optimised
    simultaneously.  Warp parameters are bounded by per-warp-type
    imperceptibility thresholds via their internal tanh() bounds.

    Args:
        image: (1, C, H, W) original image in [0, 1].
        vae_optimiser: VAELatentOptimiser instance (frozen).
        warps: dict of {warp_type: warp_module} or single warp module.
            Warp parameters must have requires_grad=True.
        warp_type: specific warp type or 'all' to cycle.
        num_iterations: Maximum optimisation iterations.
        learning_rate: Learning rate for Adam (optimises Δz + θ_warp).
        landmark_weight: λ_lm — weight of landmark-drift loss (monitor only).
        latent_reg_weight: λ_reg — weight of latent regularisation ||Δz||².
        ssim_weight: λ_ssim — weight of differentiable SSIM preservation loss.
        pixel_weight: λ_pix — weight of differentiable pixel L2 preservation loss.
        epsilon: Max allowed landmark displacement (pixels) for convergence.
        device: Device to run on.
        save_dir: Directory for per-iteration images.
        save_interval: Save image every N iterations.
        verbose: Print per-iteration loss values.
        whole_image: Apply warp to entire image (not just face mask).
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Step 1: Encode original image ──────────────────────────
    print("Encoding original image to VAE latent space ...")
    with torch.no_grad():
        z_orig = vae_optimiser.encode(image)   # (1, C, H', W') frozen
    print(f"  Latent shape: {z_orig.shape}")

    # ── Learnable latent delta ─────────────────────────────────
    delta_z = nn.Parameter(torch.zeros_like(z_orig), requires_grad=True)

    # ── Detect original landmarks ──────────────────────────────
    print("Detecting landmarks on original image ...")
    with torch.no_grad():
        P_orig = landmark_detector.detect(image)  # (1, N, 2)

    # ── Original embedding ─────────────────────────────────────
    with torch.no_grad():
        E_orig = extract_embedding(identity_model, image)

    # ── Determine warp list ────────────────────────────────────
    if warp_type == "all":
        warp_keys = list(WARP_TYPES.keys())
        print(f"Cycling through {len(warp_keys)} warp types per iteration.")
    else:
        warp_keys = [warp_type]

    # ── Set face masks for all warps ───────────────────────────
    if not whole_image:
        for wk in warp_keys:
            w = warps[wk] if isinstance(warps, dict) else warps
            if hasattr(w, 'set_face_mask'):
                face_mask = create_face_mask(
                    landmarks=P_orig[0], image_size=image.shape[2:],
                    padding=0.15, blur_sigma=8.0, device=device,
                )
                w.set_face_mask(face_mask)

    # ── Collect all learnable parameters: Δz + warp params ─────
    all_params = [delta_z]
    for wk in warp_keys:
        w = warps[wk] if isinstance(warps, dict) else warps
        for p in w.parameters():
            if p.requires_grad:
                all_params.append(p)

    num_warp_params = len(all_params) - 1
    print(f"  Learnable parameters: 1 (Δz) + {num_warp_params} (warp θ)")

    # ── Joint optimiser over Δz + warp params ──────────────────
    optimizer = torch.optim.Adam(all_params, lr=learning_rate)

    history = []
    best_loss = float("inf")
    best_dz = None
    best_warp_states = {}

    print(f"\nRunning joint VAE latent + warp optimisation for {num_iterations} iterations ...")
    print(f"  lr={learning_rate}  λ_lm={landmark_weight}  λ_reg={latent_reg_weight}"
          f"  λ_ssim={ssim_weight}  λ_pix={pixel_weight}")
    print(f"  ε={epsilon}px  (imperceptibility thresholds enforced via tanh bounds)")
    print(f"  Warp: {warp_type}")
    print(f"  Saving images every {save_interval} iterations → {save_dir}\n")


    # Save original
    save_image(image[0], os.path.join(save_dir, "iter_0000_original.png"))

    pbar = tqdm(range(num_iterations), desc="Joint VAE+warp optimisation")
    for iteration in pbar:
        # ── Step 2: Decode perturbed image from latent ─────────
        z_perturbed = z_orig + delta_z
        perturbed = vae_optimiser.decode(z_perturbed)  # (1,3,H,W) gradient flows

        # ── Step 3: Apply geometric warp (cycle if 'all') ──────
        # Warp parameters are learnable; gradient flows through warp
        # to both perturbed (→Δz) and warp params (θ_warp)
        if warp_type == "all":
            wk = warp_keys[iteration % len(warp_keys)]
            w = warps[wk]
        else:
            wk = warp_type
            w = warps[wk] if isinstance(warps, dict) else warps

        warped = w(perturbed)

        # ── Step 4: Detect landmarks on warped image ───────────
        # MediaPipe is non-differentiable → no gradient to Δz or θ_warp
        # L_landmark serves as a monitoring metric only
        with torch.no_grad():
            P_pert = landmark_detector.detect(warped.detach())

        # ── Step 5: Extract embedding (gradient flows to Δz + θ_warp) ──
        E_pert = extract_embedding(identity_model, warped)

        # ── Step 6: Compute losses ─────────────────────────────
        # Identity loss (minimise to disrupt identity)
        L_id = identity_loss(E_orig, E_pert)

        # Landmark loss (non-diff, monitor only — no gradient)
        L_lm = landmark_loss(P_orig, P_pert)

        # Latent regularisation (differentiable → Δz)
        L_reg = latent_regularisation(delta_z)

        # SSIM preservation on BOTH perturbed and warped images:
        # - perturbed vs original: constrains Δz (latent perturbation)
        # - warped vs perturbed: constrains θ_warp (geometric displacement)
        L_ssim_pert = ssim_loss(perturbed, image)           # differentiable → Δz
        L_ssim_warp = ssim_loss(warped, perturbed)          # differentiable → Δz + θ_warp
        L_ssim = L_ssim_pert + L_ssim_warp

        # Pixel L2 preservation on BOTH perturbed and warped:
        L_pix_pert = pixel_l2_loss(perturbed, image)        # differentiable → Δz
        L_pix_warp = pixel_l2_loss(warped, perturbed)       # differentiable → Δz + θ_warp
        L_pix = L_pix_pert + L_pix_warp

        L_total = (L_id
                   + ssim_weight * L_ssim
                   + pixel_weight * L_pix
                   + landmark_weight * L_lm
                   + latent_reg_weight * L_reg)


        # ── Step 7: Backprop & update Δz + θ_warp ─────────────
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        # Track best (save both Δz and warp parameter states)
        loss_info = {
            "L_identity": L_id.item(),
            "L_landmark": L_lm.item(),
            "L_latent_reg": L_reg.item(),
            "L_ssim": L_ssim.item(),
            "L_pixel": L_pix.item(),
            "L_total": L_total.item(),
            "warp": wk,
        }
        if L_total.item() < best_loss:
            best_loss = L_total.item()
            best_dz = delta_z.data.clone()
            # Save warp parameter states
            best_warp_states = {}
            for wk_save in warp_keys:
                w_save = warps[wk_save] if isinstance(warps, dict) else warps
                best_warp_states[wk_save] = copy.deepcopy(w_save.state_dict())

        history.append({"iteration": iteration, **loss_info})
        pbar.set_postfix({
            "L_total": f"{loss_info['L_total']:.4f}",
            "L_id": f"{loss_info['L_identity']:.4f}",
            "L_ssim": f"{loss_info['L_ssim']:.4f}",
            "warp": wk,
        })
        if verbose and iteration % 10 == 0:
            warp_mag = get_warp_magnitude(w, wk)
            print(
                f"  Iter {iteration:4d} [{wk:10s}] | "
                f"L_total={loss_info['L_total']:.6f}  "
                f"L_id={loss_info['L_identity']:.6f}  "
                f"L_ssim={loss_info['L_ssim']:.6f}  "
                f"L_pix={loss_info['L_pixel']:.6f}  "
                f"L_lm={loss_info['L_landmark']:.6f}  "
                f"|warp|={warp_mag:.4f}"
            )


        # ── Save images periodically ───────────────────────────
        if iteration % save_interval == 0:
            img_path = os.path.join(save_dir, f"iter_{iteration:04d}.png")
            save_image(perturbed[0].detach(), img_path)
            warped_path = os.path.join(save_dir, f"iter_{iteration:04d}_warped.png")
            save_image(warped[0].detach(), warped_path)

        # ── Stopping criteria ──────────────────────────────────
        identity_disrupted = loss_info["L_identity"] < 0.5
        landmarks_preserved = loss_info["L_landmark"] < epsilon
        if identity_disrupted and landmarks_preserved:
            print(f"\n✓ Converged at iteration {iteration}: "
                  f"identity_sim={loss_info['L_identity']:.4f}  "
                  f"landmark_drift={loss_info['L_landmark']:.4f}")
            break

    # Save final images
    final_iter = history[-1]["iteration"] if history else 0
    if final_iter % save_interval != 0:
        save_image(perturbed[0].detach(),
                   os.path.join(save_dir, f"iter_{final_iter:04d}_final.png"))
        save_image(warped[0].detach(),
                   os.path.join(save_dir, f"iter_{final_iter:04d}_warped_final.png"))

    # ── Restore best Δz AND warp parameter states ──────────────
    if best_dz is not None:
        delta_z.data = best_dz
    if best_warp_states:
        for wk_restore in warp_keys:
            w_restore = warps[wk_restore] if isinstance(warps, dict) else warps
            if wk_restore in best_warp_states:
                w_restore.load_state_dict(best_warp_states[wk_restore])

    with torch.no_grad():
        z_final = z_orig + delta_z
        final_perturbed = vae_optimiser.decode(z_final)

    # Apply each warp (with optimised params) to the final perturbed image
    final_warped = {}
    if warp_type == "all":
        for wk in WARP_TYPES:
            w = warps[wk]
            final_warped[wk] = w(final_perturbed).detach()
    else:
        w = warps[warp_type] if isinstance(warps, dict) else warps
        final_warped[warp_type] = w(final_perturbed).detach()

    return final_perturbed.detach(), final_warped, history



# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="VAE latent optimisation for adversarial geometric perturbations"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output", "-o", type=str,
                        default="results/vae_latent_out.png",
                        help="Path to save the perturbed output image")
    parser.add_argument("--warp-type", type=str, default="bspline",
                        choices=list(WARP_TYPES.keys()) + ["all"],
                        help="Geometric warp type (or 'all' to cycle)")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Maximum optimisation iterations")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for Adam (optimises Δz)")
    parser.add_argument("--landmark-weight", type=float, default=0.1,
                        help="λ_lm — weight of landmark-drift loss (monitor only, non-diff)")
    parser.add_argument("--latent-reg-weight", type=float, default=0.001,
                        help="λ_reg — weight of latent regularisation ||Δz||²")
    parser.add_argument("--ssim-weight", type=float, default=5.0,
                        help="λ_ssim — weight of differentiable SSIM preservation loss")
    parser.add_argument("--pixel-weight", type=float, default=1.0,
                        help="λ_pix — weight of differentiable pixel L2 preservation loss")
    parser.add_argument("--epsilon", type=float, default=10.0,
                        help="Max allowed landmark displacement (pixels)")
    parser.add_argument("--grid-size", type=int, nargs=2, default=[8, 8],
                        help="Control point grid (rows cols)")
    parser.add_argument("--imperceptibility-scale", type=float, default=1.0,
                        help="Scale factor for per-warp-type imperceptibility thresholds "
                             "(1.0=conservative, 2.0=looser, 0.5=tighter)")

    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512],
                        help="Image resize dimensions (H W)")
    parser.add_argument("--model-id", type=str,
                        default="timbrooks/instruct-pix2pix",
                        help="HuggingFace model ID for InstructPix2Pix VAE")
    parser.add_argument("--taesd-path", type=str, default="taesd",
                        help="Path to local TAESD directory (fallback VAE)")
    parser.add_argument("--target-model", type=str, default="facenet",
                        choices=["facenet", "arcface"],
                        help="Identity model for embedding extraction")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str,
                        default="results/vae_latent_iterations",
                        help="Directory for per-iteration images")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save image every N iterations")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate final comparison visualisation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-iteration loss values")
    parser.add_argument("--whole-image", action="store_true",
                        help="Apply the warp to the entire image")
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    image_size = tuple(args.image_size)

    # ── Load original image ────────────────────────────────────
    print(f"Loading image: {args.input}")
    image = load_image(args.input, size=image_size, device=device)
    image = image.unsqueeze(0)  # (1, C, H, W)
    print(f"  Image shape: {image.shape}")

    # ── Initialise VAE latent optimiser ────────────────────────
    print("\nInitialising VAE ...")
    vae_optimiser = VAELatentOptimiser(
        model_id=args.model_id,
        taesd_path=args.taesd_path,
        device=device,
    )
    # Freeze VAE — only Δz is learnable
    for p in vae_optimiser.parameters():
        p.requires_grad = False
    vae_optimiser.eval()

    # ── Initialise warp(s) with learnable params + imperceptibility thresholds ─
    grid_size = tuple(args.grid_size)
    if args.warp_type == "all":
        print(f"Initialising all {len(WARP_TYPES)} warp types (learnable, imperceptible) ...")
        warps = create_all_warps(image_size, grid_size, device,
                                 imperceptibility_scale=args.imperceptibility_scale)
    else:
        print(f"Initialising {args.warp_type} warp (learnable, imperceptible) ...")
        warps = create_warp(args.warp_type, image_size, grid_size,
                            imperceptibility_scale=args.imperceptibility_scale).to(device)
    # Warp parameters are already requires_grad=True from create_warp()
    thresholds = IMPERCEPTIBILITY_THRESHOLDS.get(args.warp_type, {})
    print(f"  Imperceptibility thresholds: {thresholds}")


    # ── Initialise identity model ──────────────────────────────
    print(f"Loading identity model: {args.target_model}")
    identity_model = FaceRecognitionModel(
        model_name=args.target_model, device=device,
    )

    # ── Initialise MediaPipe landmark detector ─────────────────
    print("Initialising MediaPipe landmark detector ...")
    landmark_detector = LandmarkDetector(
        detector_type="mediapipe", num_landmarks=68, device=device,
    )

    # ── Run VAE latent optimisation ────────────────────────────
    perturbed, final_warped, history = run_vae_latent_optimisation(
        image=image,
        vae_optimiser=vae_optimiser,
        warps=warps,
        identity_model=identity_model,
        landmark_detector=landmark_detector,
        warp_type=args.warp_type,
        num_iterations=args.iterations,
        learning_rate=args.lr,
        landmark_weight=args.landmark_weight,
        latent_reg_weight=args.latent_reg_weight,
        ssim_weight=args.ssim_weight,
        pixel_weight=args.pixel_weight,
        epsilon=args.epsilon,
        device=device,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        verbose=args.verbose,
        whole_image=args.whole_image,
    )


    # ── Save final perturbed image ─────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_image(perturbed[0], args.output)
    print(f"\nSaved perturbed image → {args.output}")

    # ── Save raw difference image (8x brightness enhanced) ─────
    # Compute difference: perturbed - original, then enhance by 8x
    with torch.no_grad():
        # Both images are in [0, 1] range
        diff = perturbed[0] - image[0]  # Difference in [-1, 1]
        # Enhance brightness by 8x
        diff_enhanced = diff * 8.0
        # Shift to [0, 1] range for visualization (0.5 = no difference)
        diff_vis = (diff_enhanced + 1.0) / 2.0
        diff_vis = diff_vis.clamp(0, 1)
        # Save difference image
        diff_output = os.path.join(os.path.dirname(args.output), "raw_difference.png")
        save_image(diff_vis, diff_output)
        print(f"Saved raw difference image (8x enhanced) → {diff_output}")

    # ── Final metrics ──────────────────────────────────────────
    with torch.no_grad():
        E_orig = extract_embedding(identity_model, image)
        E_pert = extract_embedding(identity_model, perturbed)
        P_orig = landmark_detector.detect(image)
        P_pert = landmark_detector.detect(perturbed)

        final_sim = F.cosine_similarity(E_orig, E_pert, dim=1).item()
        final_drift = torch.norm(P_pert - P_orig, dim=2).mean().item()

    print("\n" + "=" * 60)
    print(f"VAE Latent Optimisation — Final Results")
    print(f"  VAE type: {vae_optimiser.vae_type}")
    print(f"  Warp type: {args.warp_type}")
    print("=" * 60)
    print(f"  Identity similarity (cos):  {final_sim:.6f}")
    print(f"  Landmark drift (px):        {final_drift:.6f}")
    print(f"  ε threshold:                {args.epsilon:.1f} px")
    print(f"  Landmarks preserved:        {'✓' if final_drift < args.epsilon else '✗'}")
    print(f"  Identity disrupted:         {'✓' if final_sim < 0.5 else '✗'}")
    print(f"  Iterations run:             {len(history)}")
    print("=" * 60)

    # ── Per-warp summary (if 'all') ────────────────────────────
    if args.warp_type == "all" and final_warped:
        print("\n── Per-Warp Summary (on optimised perturbed image) ──")
        print(f"{'Warp':<14} {'Identity Sim':<14} {'Landmark Drift':<16} {'Disrupted'}")
        print("-" * 58)
        for wk, warped_img in final_warped.items():
            with torch.no_grad():
                E_w = extract_embedding(identity_model, warped_img)
                P_w = landmark_detector.detect(warped_img)
                sim_w = F.cosine_similarity(E_orig, E_w, dim=1).item()
                drift_w = torch.norm(P_w - P_orig, dim=2).mean().item()
            disrupted = "✓" if sim_w < 0.5 else "✗"
            print(f"{wk:<14} {sim_w:<14.6f} {drift_w:<16.6f} {disrupted}")

    # ── Optional visualisation ─────────────────────────────────
    if args.visualize:
        print("\nGenerating visualisations ...")
        viz_dir = os.path.dirname(args.output) or "."
        viz = Visualizer(save_dir=viz_dir)

        # Comparison: original vs perturbed
        viz.visualize_comparison(
            image[0], perturbed[0],
            title="Original vs VAE Latent Perturbed",
            save_path="vae_latent_comparison.png",
        )
        viz.visualize_perturbation_magnitude(
            image[0], perturbed[0],
            title="VAE Latent Perturbation Magnitude",
            save_path="vae_latent_perturbation_magnitude.png",
        )

        # If single warp, also show original vs warped
        if args.warp_type != "all":
            warped_final = final_warped[args.warp_type]
            viz.visualize_comparison(
                image[0], warped_final[0],
                title=f"Original vs {args.warp_type} Warped (VAE latent)",
                save_path=f"vae_latent_{args.warp_type}_warped_comparison.png",
            )

        # Loss curves
        import matplotlib.pyplot as plt
        if history:
            iters = [h["iteration"] for h in history]
            fig, axes = plt.subplots(2, 3, figsize=(18, 8))
            loss_keys = ["L_identity", "L_landmark", "L_latent_reg",
                         "L_ssim", "L_pixel", "L_total"]
            for ax, key in zip(axes.flat, loss_keys):
                vals = [h[key] for h in history]
                ax.plot(iters, vals, linewidth=2)
                ax.set_title(key)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss")
                ax.grid(True, alpha=0.3)
            plt.suptitle("Joint VAE Latent + Warp Optimisation — Loss Curves")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "vae_latent_loss_curves.png"), dpi=150)

            print(f"  Saved loss curves → {viz_dir}/vae_latent_loss_curves.png")

        viz.close_all()

    print("\nDone!")


if __name__ == "__main__":
    main()
