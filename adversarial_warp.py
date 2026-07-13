#!/usr/bin/env python
"""
Adversarial Geometric Warp via Landmark-Constrained Identity Disruption.

This script implements the following pipeline:

    1. Input  : Original Image → Get P_orig via MediaPipe.
    2. Transform: Apply a warp (B-spline / TPS) using parameters θ.
    3. Detect : Run the warped image through MediaPipe to get P_pert.
    4. Embed   : Run the warped image through an Identity Model to get E_pert.
    5. Calculate: Compute L_total = L_identity + λ · L_landmark.
    6. Optimize: Use Adam to update θ to reduce L_total.
    7. Repeat  : Iterate until identity is disrupted but landmarks
                 haven't moved beyond the ε threshold.

Usage:
    python adversarial_warp.py --input original.jpg --output results/warped_output.png
    python adversarial_warp.py -i original.jpg -o results/warped.png --iterations 300 --lr 0.01
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from config import Config
from transforms import BSplineWarp, create_face_mask
from models import FaceRecognitionModel
from utils import load_image, save_image, Visualizer
from utils.landmarks import LandmarkDetector



# ──────────────────────────────────────────────────────────────
#  Embedding extraction (with gradient flow)
# ──────────────────────────────────────────────────────────────

def extract_embedding(identity_model: FaceRecognitionModel, x: torch.Tensor) -> torch.Tensor:
    """
    Extract identity embedding from an image, preserving gradient flow.

    The FaceRecognitionModel.forward() wraps the model call in
    torch.no_grad(), which would block gradients from reaching the warp
    parameters θ.  This helper calls the underlying model directly so
    that gradients flow through the (frozen) identity model back to θ.

    Parameters
    ----------
    identity_model : FaceRecognitionModel
        Frozen face-recognition model (params have requires_grad=False).
    x : (B, C, H, W) tensor
        Input image (must have requires_grad=True for the warp params).

    Returns
    -------
    embedding : (B, D) tensor
        L2-normalised identity embedding with gradient flow.
    """
    # Resize to the model's expected input size (differentiable)
    if x.shape[2:] != identity_model.input_size:
        x = F.interpolate(
            x, size=identity_model.input_size,
            mode='bilinear', align_corners=False
        )

    # Forward through the underlying model — NO torch.no_grad() here
    # so gradients flow to the warp parameters θ.
    # Model params themselves are frozen (requires_grad=False).
    embedding = identity_model.model(x)

    # L2-normalise
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding


# ──────────────────────────────────────────────────────────────
#  Loss helpers
# ──────────────────────────────────────────────────────────────

def landmark_loss(P_orig: torch.Tensor, P_pert: torch.Tensor) -> torch.Tensor:
    """
    L_landmark = mean ||P_pert - P_orig||₂

    Measures how far the facial landmarks have drifted.
    """
    displacement = P_pert - P_orig                      # (B, N, 2)
    magnitude = torch.norm(displacement, dim=2)          # (B, N)
    return magnitude.mean()


def identity_loss(E_orig: torch.Tensor, E_pert: torch.Tensor) -> torch.Tensor:
    """
    L_identity = cos(E_orig, E_pert)

    Cosine similarity between identity embeddings.  Minimising this
    pushes the perturbed embedding away from the original → identity
    disruption.
    """
    return F.cosine_similarity(E_orig, E_pert, dim=1).mean()


def total_loss(
    E_orig: torch.Tensor,
    E_pert: torch.Tensor,
    P_orig: torch.Tensor,
    P_pert: torch.Tensor,
    landmark_weight: float,
) -> tuple[torch.Tensor, dict]:
    """
    L_total = L_identity + λ · L_landmark

    Returns the scalar total loss plus a dict of individual components
    for logging.
    """
    L_id = identity_loss(E_orig, E_pert)
    L_lm = landmark_loss(P_orig, P_pert)
    L_total = L_id + landmark_weight * L_lm

    return L_total, {
        "L_identity": L_id.item(),
        "L_landmark": L_lm.item(),
        "L_total": L_total.item(),
    }


# ──────────────────────────────────────────────────────────────
#  Main optimisation loop
# ──────────────────────────────────────────────────────────────

def run_adversarial_warp(
    image: torch.Tensor,
    warp: BSplineWarp,
    identity_model: FaceRecognitionModel,
    landmark_detector: LandmarkDetector,
    *,
    num_iterations: int = 500,
    learning_rate: float = 0.01,
    landmark_weight: float = 1.0,
    epsilon: float = 5.0,
    device: str = "cuda",
    save_dir: str = "results/warp_iterations",
    save_interval: int = 50,
    verbose: bool = True,
) -> tuple[torch.Tensor, list]:
    """
    Run the adversarial warp optimisation loop.

    Parameters
    ----------
    image : (1, C, H, W) tensor
        Original input image.
    warp : BSplineWarp
        Differentiable B-spline warp whose parameters θ will be optimised.
        The warp is constrained to the face region via a face mask.
    identity_model : FaceRecognitionModel
        Frozen identity / face-recognition model used for E_orig, E_pert.
    landmark_detector : LandmarkDetector
        MediaPipe-based landmark detector for P_orig, P_pert.
    num_iterations : int
        Maximum optimisation steps.
    learning_rate : float
        Adam learning rate for θ.
    landmark_weight : float
        λ — weight of the landmark-drift term in L_total.
    epsilon : float
        Maximum allowed landmark displacement (pixels).  If the mean
        displacement stays below this we consider landmarks "preserved".
    device : str
        'cuda' or 'cpu'.
    save_dir : str
        Directory for per-iteration visualisations.
    save_interval : int
        Save an image every *save_interval* iterations.
    verbose : bool
        Print per-iteration loss values.

    Returns
    -------
    perturbed : (1, C, H, W) tensor
        Final warped (adversarial) image.
    history : list[dict]
        Per-iteration loss values.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Step 1: P_orig — landmarks on the original image ──────────
    with torch.no_grad():
        P_orig = landmark_detector.detect(image)          # (1, N, 2)

    # ── Create face mask from landmarks ───────────────────────────
    # This restricts the warp to the face region only.
    print("Creating face mask from landmarks ...")
    face_mask = create_face_mask(
        landmarks=P_orig[0],  # (N, 2)
        image_size=image.shape[2:],  # (H, W)
        padding=0.15,
        blur_sigma=8.0,
        device=device,
    )
    warp.set_face_mask(face_mask)
    print(f"  Face mask created: {face_mask.shape}, "
          f"coverage={face_mask.mean().item():.3f}")

    # ── E_orig — identity embedding of the original image ──────────
    with torch.no_grad():
        E_orig = extract_embedding(identity_model, image)  # (1, D)


    # ── Optimiser over warp parameters θ ───────────────────────────
    optimizer = torch.optim.Adam(warp.parameters(), lr=learning_rate)

    history = []
    best_loss = float("inf")
    best_state = None

    pbar = tqdm(range(num_iterations), desc="Adversarial warp")
    for iteration in pbar:
        # ── Step 2: Transform — apply warp with parameters θ ──────
        perturbed = warp(image)                           # (1, C, H, W)

        # ── Step 3: Detect — P_pert via MediaPipe on warped image ─
        # MediaPipe runs on CPU numpy; detach from graph.
        with torch.no_grad():
            P_pert = landmark_detector.detect(perturbed.detach())

        # ── Step 4: Embed — E_pert from identity model ─────────────
        # Use extract_embedding (not identity_model.forward) so that
        # gradients flow through the frozen model back to θ.
        E_pert = extract_embedding(identity_model, perturbed)  # (1, D)

        # ── Step 5: Calculate — L_total ────────────────────────────
        loss, loss_info = total_loss(
            E_orig, E_pert, P_orig, P_pert, landmark_weight
        )

        # ── Step 6: Optimize — backprop & update θ ────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                k: v.clone() for k, v in warp.state_dict().items()
            }

        # Logging
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

        # ── Step 7: Repeat — check stopping criteria ──────────────
        # Stop early if identity is sufficiently disrupted *and*
        # landmarks are within the ε threshold.
        identity_disrupted = loss_info["L_identity"] < 0.5   # cos sim < 0.5
        landmarks_preserved = loss_info["L_landmark"] < epsilon

        if identity_disrupted and landmarks_preserved:
            print(
                f"\n✓ Converged at iteration {iteration}: "
                f"identity_sim={loss_info['L_identity']:.4f}  "
                f"landmark_drift={loss_info['L_landmark']:.4f} < ε={epsilon}"
            )
            break

        # Periodic visualisation
        if iteration % save_interval == 0:
            img_path = os.path.join(save_dir, f"iter_{iteration:04d}.png")
            save_image(perturbed[0], img_path)

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
        description="Adversarial geometric warp: disrupt identity while preserving landmarks"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="results/warped_output.png",
                        help="Path to save the perturbed output image")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Maximum optimisation iterations")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for Adam")
    parser.add_argument("--landmark-weight", type=float, default=1.0,
                        help="λ — weight of landmark-drift loss")
    parser.add_argument("--epsilon", type=float, default=5.0,
                        help="Max allowed landmark displacement (pixels)")
    parser.add_argument("--grid-size", type=int, nargs=2, default=[8, 8],
                        help="B-spline control point grid (rows cols)")
    parser.add_argument("--max-displacement", type=float, default=10.0,
                        help="Max pixel displacement for B-spline control points")

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

    # ── Initialise B-spline warp with parameters θ ────────────────
    grid_size = tuple(args.grid_size)
    num_cp = grid_size[0] * grid_size[1]
    print(f"Initialising B-spline warp: {grid_size[0]}×{grid_size[1]} = "
          f"{num_cp} control points, "
          f"max displacement={args.max_displacement}px")
    warp = BSplineWarp(
        image_size=image_size,
        grid_size=grid_size,
        max_displacement=args.max_displacement,
        learnable=True,
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

    # ── Run the adversarial warp optimisation ──────────────────────
    print(f"\nRunning optimisation for {args.iterations} iterations ...")
    print(f"  lr={args.lr}  λ={args.landmark_weight}  ε={args.epsilon}px\n")

    perturbed, history = run_adversarial_warp(
        image=image,
        warp=warp,
        identity_model=identity_model,
        landmark_detector=landmark_detector,
        num_iterations=args.iterations,
        learning_rate=args.lr,
        landmark_weight=args.landmark_weight,
        epsilon=args.epsilon,
        device=device,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        verbose=args.verbose,
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
    print("Final Results")
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
            title="Original vs Adversarial Warp",
            save_path="warp_comparison.png",
        )
        viz.visualize_perturbation_magnitude(
            image[0], perturbed[0].detach(),
            title="Warp Perturbation Magnitude",
            save_path="warp_perturbation_magnitude.png",
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
            plt.suptitle("Adversarial Warp — Loss Curves")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "warp_loss_curves.png"), dpi=150)
            print(f"  Saved loss curves → {viz_dir}/warp_loss_curves.png")

        viz.close_all()

    print("\nDone!")


if __name__ == "__main__":
    main()
