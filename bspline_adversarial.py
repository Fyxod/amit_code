#!/usr/bin/env python
"""
Adversarial Geometric Face Warp (Multi-Transform).

A self-contained script that supports 13 differentiable geometric warp
types.  By default the warp is constrained to the face region via a
face mask; use ``--whole-image`` to apply the transform to the entire
image.

    1.  bspline        — Cubic B-spline control point displacements
    2.  polar          — Polar coordinate (r, θ) perturbation
    3.  bezier         — Bezier (Bernstein polynomial) surface warp
    4.  lens           — Lens barrel/pincushion (Brown-Conrady) distortion
    5.  mobius         — Mobius (fractional linear) conformal transform
    6.  laplacian      — Laplacian-smoothed displacement field
    7.  geodesic       — Geodesic deformation on a Riemannian manifold
    8.  diffgeom       — Differential geometry (gradient/Laplacian) warp
    9.  delaunay       — Delaunay triangulation warp
    10. tps            — Thin-Plate Spline warp
    11. rolling        — Rolling shutter effect
    12. fft            — FFT phase perturbation
    13. homography     — Homography (perspective) transform

Pipeline:
    1. Load image → detect P_orig via MediaPipe.
    2. Create face mask from landmarks (unless --whole-image).
    3. Apply warp (parameters θ) to face region (or whole image).
    4. Re-detect P_pert on warped image.
    5. Extract E_pert from frozen identity model.
    6. Compute L_total = L_identity + λ · L_landmark.
    7. Optimise θ with Adam.
    8. Iterate until identity disrupted but landmarks within ε.

Usage:
    python bspline_adversarial.py -i original.jpg --warp-type bspline
    python bspline_adversarial.py -i original.jpg --warp-type tps --whole-image
    python bspline_adversarial.py -i original.jpg --warp-type fft --visualize
"""

import argparse
import os
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
#  Warp registry
# ──────────────────────────────────────────────────────────────

WARP_TYPES = {
    "bspline": BSplineWarp,
    "polar": PolarWarp,
    "bezier": BezierWarp,
    "lens": LensDistortion,
    "mobius": MobiusWarp,
    "laplacian": LaplacianSmoothingWarp,
    "geodesic": GeodesicWarp,
    "diffgeom": DiffGeometryWarp,
    "delaunay": DelaunayWarp,
    "tps": ThinPlateSpline,
    "rolling": RollingShutter,
    "fft": FFTPhasePerturbation,
    "homography": HomographyTransform,
}


def create_warp(warp_type, image_size, max_displacement, grid_size):
    """Create a warp module of the specified type."""
    if warp_type == "bspline":
        return BSplineWarp(
            image_size=image_size,
            grid_size=grid_size,
            max_displacement=max_displacement,
        )
    elif warp_type == "polar":
        return PolarWarp(
            image_size=image_size,
            num_radial=grid_size[0],
            num_angular=grid_size[1],
            max_radial_shift=max_displacement,
            max_angular_shift=0.1,
        )
    elif warp_type == "bezier":
        return BezierWarp(
            image_size=image_size,
            degree_u=grid_size[0] - 1,
            degree_v=grid_size[1] - 1,
            max_displacement=max_displacement,
        )
    elif warp_type == "lens":
        return LensDistortion(
            image_size=image_size,
            max_k=2.0,
            max_p=0.5,
        )
    elif warp_type == "mobius":
        return MobiusWarp(
            image_size=image_size,
            max_param=0.8,
        )
    elif warp_type == "laplacian":
        return LaplacianSmoothingWarp(
            image_size=image_size,
            grid_size=grid_size,
            max_displacement=max_displacement,
            smoothing_lambda=0.5,
        )
    elif warp_type == "geodesic":
        return GeodesicWarp(
            image_size=image_size,
            num_bumps=grid_size[0] * grid_size[1],
            max_displacement=max_displacement,
        )
    elif warp_type == "diffgeom":
        return DiffGeometryWarp(
            image_size=image_size,
            num_bumps=grid_size[0] * grid_size[1],
            max_displacement=max_displacement,
        )
    elif warp_type == "delaunay":
        return DelaunayWarp(
            image_size=image_size,
            num_points=grid_size[0] * grid_size[1],
            max_displacement=max_displacement,
        )
    elif warp_type == "tps":
        return ThinPlateSpline(
            image_size=image_size,
            num_control_points=grid_size[0] * grid_size[1],
            max_displacement=max_displacement,
        )
    elif warp_type == "rolling":
        return RollingShutter(
            image_size=image_size,
            max_offset=max_displacement,
            direction='horizontal',
            wave_type='sine',
            num_harmonics=4,
        )
    elif warp_type == "fft":
        return FFTPhasePerturbation(
            image_size=image_size,
            magnitude=0.5,
            phase_resolution=grid_size,
            shared_channels=True,
        )
    elif warp_type == "homography":
        return HomographyTransform(
            image_size=image_size,
            max_perturbation=0.15,
        )
    else:
        raise ValueError(f"Unknown warp type: {warp_type}")



# ──────────────────────────────────────────────────────────────
#  Embedding extraction (with gradient flow)
# ──────────────────────────────────────────────────────────────

def extract_embedding(identity_model, x):
    """
    Extract identity embedding with gradient flow to the warp params.
    """
    if x.shape[2:] != identity_model.input_size:
        x = F.interpolate(
            x, size=identity_model.input_size,
            mode='bilinear', align_corners=False
        )
    embedding = identity_model.model(x)
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding


# ──────────────────────────────────────────────────────────────
#  Loss functions
# ──────────────────────────────────────────────────────────────

def identity_loss(E_orig, E_pert):
    """L_identity = cos(E_orig, E_pert) — minimise to disrupt identity."""
    return F.cosine_similarity(E_orig, E_pert, dim=1).mean()


def landmark_loss(P_orig, P_pert):
    """L_landmark = mean ||P_pert - P_orig||₂ — minimise to preserve landmarks."""
    displacement = P_pert - P_orig
    magnitude = torch.norm(displacement, dim=2)
    return magnitude.mean()


# ──────────────────────────────────────────────────────────────
#  Main optimisation loop
# ──────────────────────────────────────────────────────────────

def run_adversarial_warp(
    image,
    warp,
    identity_model,
    landmark_detector,
    *,
    warp_type="bspline",
    num_iterations=500,
    learning_rate=0.01,
    landmark_weight=1.0,
    epsilon=50.0,
    device="cuda",
    save_dir="results/warp_iterations",
    save_interval=50,
    verbose=True,
    whole_image=False,
):
    """
    Run the adversarial warp optimisation loop.

    By default only the face region is warped (background preserved).
    If *whole_image* is True, the transform is applied to the entire image.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Step 1: P_orig — landmarks on the original image ──────────
    print("Detecting landmarks on original image ...")
    with torch.no_grad():
        P_orig = landmark_detector.detect(image)  # (1, N, 2)

    # ── Create face mask from landmarks (unless whole-image mode) ──
    if whole_image:
        print("Whole-image mode — no face mask, transform applied to entire image.")
    elif not hasattr(warp, 'set_face_mask'):
        print(f"  Note: {warp_type} warp does not support face masking — "
              f"applying to entire image.")
    else:
        print("Creating face mask from landmarks ...")
        face_mask = create_face_mask(
            landmarks=P_orig[0],
            image_size=image.shape[2:],
            padding=0.15,
            blur_sigma=8.0,
            device=device,
        )
        warp.set_face_mask(face_mask)
        coverage = face_mask.mean().item()
        print(f"  Face mask: {face_mask.shape}, coverage={coverage:.3f}")



    # ── E_orig — identity embedding of the original image ──────────
    with torch.no_grad():
        E_orig = extract_embedding(identity_model, image)  # (1, D)

    # ── Optimiser over warp parameters θ ───────────────────────────
    optimizer = torch.optim.Adam(warp.parameters(), lr=learning_rate)

    history = []
    best_loss = float("inf")
    best_state = None

    print(f"\nRunning {warp_type} optimisation for {num_iterations} iterations ...")
    print(f"  lr={learning_rate}  λ={landmark_weight}  ε={epsilon}px")
    print(f"  Saving warped images every {save_interval} iterations → {save_dir}\n")

    # Save the original image for reference
    save_image(image[0], os.path.join(save_dir, "iter_0000_original.png"))

    pbar = tqdm(range(num_iterations), desc=f"{warp_type} warp")
    for iteration in pbar:
        # ── Step 2: Transform — apply warp ──────────────────────────
        perturbed = warp(image)  # (1, C, H, W)

        # ── Step 3: Detect — P_pert via MediaPipe on warped image ───
        with torch.no_grad():
            P_pert = landmark_detector.detect(perturbed.detach())

        # ── Step 4: Embed — E_pert from identity model ─────────────
        E_pert = extract_embedding(identity_model, perturbed)

        # ── Step 5: Calculate — L_total ────────────────────────────
        L_id = identity_loss(E_orig, E_pert)
        L_lm = landmark_loss(P_orig, P_pert)
        L_total = L_id + landmark_weight * L_lm

        # ── Step 6: Optimize — backprop & update θ ────────────────
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        # Track best
        loss_info = {
            "L_identity": L_id.item(),
            "L_landmark": L_lm.item(),
            "L_total": L_total.item(),
        }
        if L_total.item() < best_loss:
            best_loss = L_total.item()
            best_state = {k: v.clone() for k, v in warp.state_dict().items()}

        history.append({"iteration": iteration, **loss_info})
        pbar.set_postfix({
            "L_total": f"{loss_info['L_total']:.4f}",
            "L_id": f"{loss_info['L_identity']:.4f}",
            "L_lm": f"{loss_info['L_landmark']:.4f}",
        })
        if verbose and iteration % 10 == 0:
            print(
                f"  Iter {iteration:4d} | "
                f"L_total={loss_info['L_total']:.6f}  "
                f"L_identity={loss_info['L_identity']:.6f}  "
                f"L_landmark={loss_info['L_landmark']:.6f}"
            )

        # ── Save warped image every save_interval iterations ────────
        if iteration % save_interval == 0:
            img_path = os.path.join(save_dir, f"iter_{iteration:04d}.png")
            save_image(perturbed[0], img_path)
            print(f"  → Saved warped image: {img_path}")

        # ── Step 7: Repeat — check stopping criteria ──────────────
        identity_disrupted = loss_info["L_identity"] < 0.5
        landmarks_preserved = loss_info["L_landmark"] < epsilon

        if identity_disrupted and landmarks_preserved:
            print(
                f"\n✓ Converged at iteration {iteration}: "
                f"identity_sim={loss_info['L_identity']:.4f}  "
                f"landmark_drift={loss_info['L_landmark']:.4f} < ε={epsilon}"
            )
            break

    # Save the final warped image (even if not a multiple of save_interval)
    final_iter = history[-1]["iteration"] if history else 0
    if final_iter % save_interval != 0:
        final_img_path = os.path.join(save_dir, f"iter_{final_iter:04d}_final.png")
        save_image(perturbed[0], final_img_path)
        print(f"  → Saved final warped image: {final_img_path}")


    # Restore best parameters
    if best_state is not None:
        warp.load_state_dict(best_state)

    # Generate final perturbed image
    with torch.no_grad():
        final_perturbed = warp(image)

    return final_perturbed, history


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adversarial geometric face warp: disrupt identity, preserve landmarks"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="results/warp_out.png",
                        help="Path to save the perturbed output image")
    parser.add_argument("--warp-type", type=str, default="bspline",
                        choices=list(WARP_TYPES.keys()),
                        help="Type of geometric warp to use")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Maximum optimisation iterations")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for Adam")
    parser.add_argument("--landmark-weight", type=float, default=0.1,
                        help="λ — weight of landmark-drift loss")
    parser.add_argument("--epsilon", type=float, default=10.0,
                        help="Max allowed landmark displacement (pixels)")
    parser.add_argument("--grid-size", type=int, nargs=2, default=[8, 8],
                        help="Control point grid (rows cols) — used by bspline, "
                             "bezier, laplacian, geodesic, diffgeom, polar")
    parser.add_argument("--max-displacement", type=float, default=30.0,
                        help="Max pixel displacement for control points")

    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512],
                        help="Image resize dimensions (H W)")
    parser.add_argument("--target-model", type=str, default="facenet",
                        choices=["facenet", "arcface"],
                        help="Identity model for embedding extraction")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="results/warp_iterations",
                        help="Directory for per-iteration images")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save image every N iterations")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate final comparison visualisation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-iteration loss values")
    parser.add_argument("--whole-image", action="store_true",
                        help="Apply the warp to the entire image instead of "
                             "only the face region")
    return parser.parse_args()



def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    image_size = tuple(args.image_size)

    # ── Load original image ────────────────────────────────────────
    print(f"Loading image: {args.input}")
    image = load_image(args.input, size=image_size, device=device)
    image = image.unsqueeze(0)  # (1, C, H, W)
    print(f"  Image shape: {image.shape}")

    # ── Initialise warp ─────────────────────────────────────────────
    grid_size = tuple(args.grid_size)
    print(f"Initialising {args.warp_type} warp: "
          f"grid={grid_size}, max displacement={args.max_displacement}px")
    warp = create_warp(
        warp_type=args.warp_type,
        image_size=image_size,
        max_displacement=args.max_displacement,
        grid_size=grid_size,
    ).to(device)

    # ── Initialise identity model ──────────────────────────────────
    print(f"Loading identity model: {args.target_model}")
    identity_model = FaceRecognitionModel(
        model_name=args.target_model,
        device=device,
    )

    # ── Initialise MediaPipe landmark detector ────────────────────
    print("Initialising MediaPipe landmark detector ...")
    landmark_detector = LandmarkDetector(
        detector_type="mediapipe",
        num_landmarks=68,
        device=device,
    )

    # ── Run the adversarial warp ───────────────────────────────────
    perturbed, history = run_adversarial_warp(
        image=image,
        warp=warp,
        identity_model=identity_model,
        landmark_detector=landmark_detector,
        warp_type=args.warp_type,
        num_iterations=args.iterations,
        learning_rate=args.lr,
        landmark_weight=args.landmark_weight,
        epsilon=args.epsilon,
        device=device,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        verbose=args.verbose,
        whole_image=args.whole_image,
    )


    # ── Save final perturbed image ────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_image(perturbed[0], args.output)
    print(f"\nSaved perturbed image → {args.output}")

    # ── Final metrics ──────────────────────────────────────────────
    with torch.no_grad():
        E_orig = extract_embedding(identity_model, image)
        E_pert = extract_embedding(identity_model, perturbed)
        P_orig = landmark_detector.detect(image)
        P_pert = landmark_detector.detect(perturbed)

        final_sim = F.cosine_similarity(E_orig, E_pert, dim=1).item()
        final_drift = torch.norm(P_pert - P_orig, dim=2).mean().item()

    print("\n" + "=" * 55)
    print(f"Final Results ({args.warp_type} warp)")
    print("=" * 55)
    print(f"  Identity similarity (cos):  {final_sim:.6f}")
    print(f"  Landmark drift (px):        {final_drift:.6f}")
    print(f"  ε threshold:                 {args.epsilon:.1f} px")
    print(f"  Landmarks preserved:         {'✓' if final_drift < args.epsilon else '✗'}")
    print(f"  Identity disrupted:          {'✓' if final_sim < 0.5 else '✗'}")
    print(f"  Iterations run:              {len(history)}")
    print("=" * 55)

    # ── Optional visualisation ────────────────────────────────────
    if args.visualize:
        print("\nGenerating visualisations ...")
        viz_dir = os.path.dirname(args.output) or "."
        viz = Visualizer(save_dir=viz_dir)

        viz.visualize_comparison(
            image[0], perturbed[0].detach(),
            title=f"Original vs {args.warp_type} Warp",
            save_path=f"{args.warp_type}_comparison.png",
        )
        viz.visualize_perturbation_magnitude(
            image[0], perturbed[0].detach(),
            title=f"{args.warp_type} Perturbation Magnitude",
            save_path=f"{args.warp_type}_perturbation_magnitude.png",
        )

        # Plot loss curves
        import matplotlib.pyplot as plt
        if history:
            iters = [h["iteration"] for h in history]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, key in zip(axes, ["L_identity", "L_landmark", "L_total"]):
                vals = [h[key] for h in history]
                ax.plot(iters, vals, linewidth=2)
                ax.set_title(key)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss")
                ax.grid(True, alpha=0.3)
            plt.suptitle(f"{args.warp_type} Adversarial Warp — Loss Curves")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{args.warp_type}_loss_curves.png"), dpi=150)
            print(f"  Saved loss curves → {viz_dir}/{args.warp_type}_loss_curves.png")

        viz.close_all()

    print("\nDone!")


if __name__ == "__main__":
    main()
