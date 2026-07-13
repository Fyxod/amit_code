#!/usr/bin/env python
"""
Visualization of Geometric Perturbations Over Iterations.

Loads checkpoints and generates visualizations showing how each geometric
perturbation evolves over iterations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse
import glob
from collections import defaultdict

from config import Config
from optimization import AdversarialOptimizer
from utils import load_image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize geometric perturbations over iterations'
    )
    parser.add_argument('--input', '-i', type=str, default='original.jpg')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--output-dir', type=str, default='results/iteration_vis')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dpi', type=int, default=150)
    return parser.parse_args()


def load_checkpoints(checkpoint_dir):
    """Load all checkpoints sorted by iteration number."""
    checkpoint_paths = sorted(
        glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pt')),
        key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
    )
    checkpoints = []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        iteration = ckpt['iteration']
        checkpoints.append((iteration, ckpt))
        print(f"  Loaded checkpoint: iteration {iteration}")
    return checkpoints


def create_optimizer(device='cpu'):
    """Create an AdversarialOptimizer with default config."""
    config = Config()
    config.device = device
    config.save_checkpoints = False
    config.early_stopping = False
    optimizer = AdversarialOptimizer(config, target_model=None, device=device)
    return optimizer, config


def apply_checkpoint_transforms(optimizer, ckpt, image, device='cpu'):
    """Load checkpoint state and apply transforms to get perturbed image."""
    state_dict = ckpt['model_state_dict']
    transform_state = {
        k: v for k, v in state_dict.items()
        if k.startswith(('fft_transform', 'delaunay_transform',
                         'homography_transform', 'tps_transform',
                         'rolling_shutter_transform'))
    }
    full_state = optimizer.state_dict()
    for k, v in transform_state.items():
        if k in full_state:
            full_state[k] = v
    optimizer.load_state_dict(full_state, strict=False)
    optimizer.to(device)
    optimizer.eval()
    with torch.no_grad():
        perturbed = optimizer.apply_transforms(image)
    return perturbed


def extract_transform_magnitudes(ckpt):
    """Extract per-transform magnitudes from a checkpoint."""
    sd = ckpt['model_state_dict']
    m = {}
    p = sd.get('fft_transform.phase_param')
    if p is not None:
        phase = torch.tanh(p) * 0.5
        m['fft_mean'] = phase.abs().mean().item()
        m['fft_max'] = phase.abs().max().item()
    p = sd.get('delaunay_transform.displacement')
    if p is not None:
        b = torch.tanh(p) * 10.0
        m['delaunay_mean'] = b.norm(dim=1).mean().item()
        m['delaunay_max'] = b.norm(dim=1).max().item()
    p = sd.get('homography_transform.homography_delta')
    if p is not None:
        b = torch.tanh(p) * 0.1
        m['homography_norm'] = b.norm().item()
    p = sd.get('tps_transform.displacement')
    if p is not None:
        b = torch.tanh(p) * 15.0
        m['tps_mean'] = b.norm(dim=1).mean().item()
        m['tps_max'] = b.norm(dim=1).max().item()
    p = sd.get('rolling_shutter_transform.amplitude')
    if p is not None:
        m['rs_amplitude'] = (torch.tanh(p) * 10.0).item()
    p = sd.get('rolling_shutter_transform.frequency')
    if p is not None:
        m['rs_frequency'] = torch.abs(p).item()
    return m


def extract_params(ckpt):
    """Extract transform parameters from a checkpoint."""
    sd = ckpt['model_state_dict']
    p = {}
    v = sd.get('fft_transform.phase_param')
    if v is not None:
        p['fft_phase'] = torch.tanh(v) * 0.5
    v = sd.get('delaunay_transform.displacement')
    if v is not None:
        p['delaunay_raw'] = v
        p['delaunay_base'] = sd.get('delaunay_transform.base_points_tensor')
    v = sd.get('homography_transform.homography_delta')
    if v is not None:
        p['homo_raw'] = v
    v = sd.get('tps_transform.displacement')
    if v is not None:
        p['tps_raw'] = v
        p['tps_ctrl'] = sd.get('tps_transform.control_points')
    amp = sd.get('rolling_shutter_transform.amplitude')
    if amp is not None:
        p['rs_amp'] = (torch.tanh(amp) * 10.0).item()
        p['rs_freq'] = torch.abs(sd['rolling_shutter_transform.frequency']).item()
        p['rs_phase'] = sd['rolling_shutter_transform.phase'].item()
        amp2 = sd.get('rolling_shutter_transform.amplitude2')
        if amp2 is not None:
            p['rs_amp2'] = (torch.tanh(amp2) * 5.0).item()
            p['rs_freq2'] = torch.abs(sd['rolling_shutter_transform.frequency2']).item()
            p['rs_phase2'] = sd['rolling_shutter_transform.phase2'].item()
    return p


def tensor_to_numpy(tensor):
    """Convert a tensor to numpy for visualization."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr, 0, 1)


def compute_rs_field(params, h=224, w=224):
    """Compute rolling shutter displacement field."""
    scan = torch.linspace(0, 1, h).unsqueeze(1).expand(h, w)
    amp = params.get('rs_amp', 0)
    freq = params.get('rs_freq', 1.0)
    phase = params.get('rs_phase', 0) * 2 * np.pi
    amp2 = params.get('rs_amp2', 0)
    freq2 = params.get('rs_freq2', 2.0)
    phase2 = params.get('rs_phase2', 0) * 2 * np.pi
    d = amp * torch.sin(2 * np.pi * freq * scan + phase)
    d = d + amp2 * torch.sin(2 * np.pi * freq2 * scan + phase2)
    return d.numpy()


def vis_perturbed_grid(image, ckpts, opt, out_dir, device='cpu', dpi=150):
    """Grid of perturbed images and difference maps at different iterations."""
    n = len(ckpts)
    if n == 0:
        return
    idx = np.linspace(0, n - 1, min(n, 8), dtype=int)
    sel = [ckpts[i] for i in idx]
    ns = len(sel)
    fig, axes = plt.subplots(2, ns, figsize=(3.5 * ns, 7))
    if ns == 1:
        axes = axes.reshape(2, 1)
    orig = tensor_to_numpy(image)
    for c, (it, ckpt) in enumerate(sel):
        pert = apply_checkpoint_transforms(opt, ckpt, image, device)
        pn = tensor_to_numpy(pert)
        axes[0, c].imshow(pn)
        axes[0, c].set_title(f'Iter {it}', fontsize=11)
        axes[0, c].axis('off')
        diff = np.abs(orig - pn)
        diff = diff / (diff.max() + 1e-8)
        axes[1, c].imshow(diff)
        axes[1, c].set_title('|Diff| (amp)', fontsize=10)
        axes[1, c].axis('off')
    axes[0, 0].set_ylabel('Perturbed', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Difference', fontsize=12, fontweight='bold')
    plt.suptitle('Geometric Perturbations Over Iterations', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'perturbation_grid.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    print("  Saved: perturbation_grid.png")


def vis_magnitude_curves(all_mag, iters, out_dir, dpi=150):
    """Plot transform magnitude curves over iterations."""
    if not all_mag:
        return
    metrics = defaultdict(list)
    for m in all_mag:
        for k, v in m.items():
            metrics[k].append(v)
    groups = {
        'Spatial Displacement (px)': [('delaunay_mean', 'Delaunay'), ('tps_mean', 'TPS')],
        'Max Displacement (px)': [('delaunay_max', 'Delaunay'), ('tps_max', 'TPS')],
        'FFT Phase': [('fft_mean', 'Mean'), ('fft_max', 'Max')],
        'Homography Norm': [('homography_norm', 'H delta')],
        'Rolling Shutter': [('rs_amplitude', 'Amplitude')],
    }
    ng = len(groups)
    fig, axes = plt.subplots(1, ng, figsize=(4 * ng, 4))
    if ng == 1:
        axes = [axes]
    for ax, (title, keys) in zip(axes, groups.items()):
        for key, label in keys:
            if key in metrics:
                ax.plot(iters, metrics[key], lw=2, marker='o', ms=4, label=label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Magnitude')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Transform Magnitudes Over Iterations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'transform_magnitudes.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    print("  Saved: transform_magnitudes.png")


def vis_loss_curves(history, out_dir, dpi=150):
    """Plot loss curves over iterations."""
    if not history:
        return
    iters = []
    components = defaultdict(list)
    for h in history:
        iters.append(h['iteration'])
        for k, v in h['losses'].items():
            components[k].append(v)
    nl = len(components)
    fig, axes = plt.subplots(1, nl, figsize=(4 * nl, 4))
    if nl == 1:
        axes = [axes]
    for ax, (k, vals) in zip(axes, components.items()):
        ax.plot(iters, vals, lw=2, marker='o', ms=3)
        ax.set_title(k.replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    plt.suptitle('Loss Curves Over Iterations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curves.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    print("  Saved: loss_curves.png")


def vis_per_transform(ckpts, out_dir, dpi=150):
    """Create per-transform detailed visualizations."""
    n = len(ckpts)
    if n == 0:
        return
    idx = np.linspace(0, n - 1, min(n, 6), dtype=int)
    sel = [ckpts[i] for i in idx]
    ns = len(sel)

    # FFT Phase
    fig, axes = plt.subplots(1, ns, figsize=(3.5 * ns, 3.5))
    if ns == 1:
        axes = [axes]
    has = False
    for c, (it, ckpt) in enumerate(sel):
        p = extract_params(ckpt)
        fp = p.get('fft_phase')
        if fp is not None:
            has = True
            axes[c].imshow(fp.mean(dim=0).numpy(), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            axes[c].set_title(f'Iter {it}', fontsize=11)
            axes[c].axis('off')
        else:
            axes[c].axis('off')
    if has:
        plt.suptitle('FFT Phase Perturbation', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fft_phase_evolution.png'), dpi=dpi, bbox_inches='tight')
        print("  Saved: fft_phase_evolution.png")
    plt.close()

    # Delaunay
    fig, axes = plt.subplots(1, ns, figsize=(3.5 * ns, 3.5))
    if ns == 1:
        axes = [axes]
    has = False
    for c, (it, ckpt) in enumerate(sel):
        p = extract_params(ckpt)
        dr = p.get('delaunay_raw')
        bp = p.get('delaunay_base')
        if dr is not None and bp is not None:
            has = True
            bd = torch.tanh(dr) * 10.0
            disp = bp + bd
            bn = bp.numpy()
            dn = disp.numpy()
            axes[c].scatter(bn[:, 0], bn[:, 1], c='blue', s=30, alpha=0.7, label='Original')
            axes[c].scatter(dn[:, 0], dn[:, 1], c='red', s=30, alpha=0.7, label='Displaced')
            for i in range(len(bn)):
                axes[c].annotate('', xy=dn[i], xytext=bn[i],
                                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
            axes[c].set_xlim(-10, 234)
            axes[c].set_ylim(234, -10)
            axes[c].set_aspect('equal')
            axes[c].set_title(f'Iter {it}', fontsize=11)
            if c == 0:
                axes[c].legend(fontsize=8)
        else:
            axes[c].axis('off')
    if has:
        plt.suptitle('Delaunay Control Point Displacements', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'delaunay_evolution.png'), dpi=dpi, bbox_inches='tight')
        print("  Saved: delaunay_evolution.png")
    plt.close()

    # Homography
    fig, axes = plt.subplots(1, ns, figsize=(3.5 * ns, 3.5))
    if ns == 1:
        axes = [axes]
    has = False
    for c, (it, ckpt) in enumerate(sel):
        p = extract_params(ckpt)
        hr = p.get('homo_raw')
        if hr is not None:
            has = True
            delta = torch.tanh(hr) * 0.1
            H = torch.eye(3)
            H[0, 0] = 1 + delta[0]; H[0, 1] = delta[1]; H[0, 2] = delta[2]
            H[1, 0] = delta[3]; H[1, 1] = 1 + delta[4]; H[1, 2] = delta[5]
            H[2, 0] = delta[6]; H[2, 1] = delta[7]
            H_inv = torch.inverse(H)
            n_lines = 17
            yc = torch.linspace(-1, 1, 224)
            xc = torch.linspace(-1, 1, 224)
            for val in torch.linspace(-1, 1, n_lines):
                pts = torch.stack([xc, torch.full((224,), val), torch.ones(224)], dim=1)
                w = (H_inv @ pts.T).T
                w = w[:, :2] / (w[:, 2:3] + 1e-8)
                axes[c].plot(w[:, 0].numpy(), w[:, 1].numpy(), 'b-', alpha=0.5, lw=0.5)
            for val in torch.linspace(-1, 1, n_lines):
                pts = torch.stack([torch.full((224,), val), yc, torch.ones(224)], dim=1)
                w = (H_inv @ pts.T).T
                w = w[:, :2] / (w[:, 2:3] + 1e-8)
                axes[c].plot(w[:, 0].numpy(), w[:, 1].numpy(), 'b-', alpha=0.5, lw=0.5)
            axes[c].set_xlim(-1.2, 1.2)
            axes[c].set_ylim(1.2, -1.2)
            axes[c].set_aspect('equal')
            axes[c].set_title(f'Iter {it}', fontsize=11)
        else:
            axes[c].axis('off')
    if has:
        plt.suptitle('Homography Warped Grid', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'homography_evolution.png'), dpi=dpi, bbox_inches='tight')
        print("  Saved: homography_evolution.png")
    plt.close()

    # TPS
    fig, axes = plt.subplots(1, ns, figsize=(3.5 * ns, 3.5))
    if ns == 1:
        axes = [axes]
    has = False
    for c, (it, ckpt) in enumerate(sel):
        p = extract_params(ckpt)
        tr = p.get('tps_raw')
        tc = p.get('tps_ctrl')
        if tr is not None and tc is not None:
            has = True
            bd = torch.tanh(tr) * 15.0 / 224.0
            disp = tc + bd
            bn = tc.numpy()
            dn = disp.numpy()
            axes[c].scatter(bn[:, 0], bn[:, 1], c='blue', s=30, alpha=0.7, label='Original')
            axes[c].scatter(dn[:, 0], dn[:, 1], c='red', s=30, alpha=0.7, label='Displaced')
            for i in range(len(bn)):
                axes[c].annotate('', xy=dn[i], xytext=bn[i],
                                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
            axes[c].set_xlim(-1.3, 1.3)
            axes[c].set_ylim(1.3, -1.3)
            axes[c].set_aspect('equal')
            axes[c].set_title(f'Iter {it}', fontsize=11)
            if c == 0:
                axes[c].legend(fontsize=8)
        else:
            axes[c].axis('off')
    if has:
        plt.suptitle('TPS Control Point Displacements', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'tps_evolution.png'), dpi=dpi, bbox_inches='tight')
        print("  Saved: tps_evolution.png")
    plt.close()

    # Rolling Shutter
    fig, axes = plt.subplots(1, ns, figsize=(3.5 * ns, 3.5))
    if ns == 1:
        axes = [axes]
    has = False
    for c, (it, ckpt) in enumerate(sel):
        p = extract_params(ckpt)
        if 'rs_amp' in p:
            has = True
            field = compute_rs_field(p)
            axes[c].imshow(field, cmap='RdBu_r', vmin=-10, vmax=10, aspect='auto')
            axes[c].set_title(f'Iter {it}', fontsize=11)
            axes[c].set_xlabel('Width')
            axes[c].set_ylabel('Height')
        else:
            axes[c].axis('off')
    if has:
        plt.suptitle('Rolling Shutter Displacement (px)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'rolling_shutter_evolution.png'), dpi=dpi, bbox_inches='tight')
        print("  Saved: rolling_shutter_evolution.png")
    plt.close()


def vis_composite(image, ckpts, opt, all_mag, iters, history, out_dir, device='cpu', dpi=150):
    """Create a composite summary figure."""
    n = len(ckpts)
    if n == 0:
        return
    idx = np.linspace(0, n - 1, min(n, 5), dtype=int)
    sel = [ckpts[i] for i in idx]
    ns = len(sel)
    orig = tensor_to_numpy(image)

    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(4, max(ns, 3), figure=fig, hspace=0.4, wspace=0.25)

    # Row 1: Perturbed images
    for c, (it, ckpt) in enumerate(sel):
        ax = fig.add_subplot(gs[0, c])
        pert = apply_checkpoint_transforms(opt, ckpt, image, device)
        ax.imshow(tensor_to_numpy(pert))
        ax.set_title(f'Iter {it}', fontsize=11, fontweight='bold')
        ax.axis('off')

    # Row 2: Difference maps
    for c, (it, ckpt) in enumerate(sel):
        ax = fig.add_subplot(gs[1, c])
        pert = apply_checkpoint_transforms(opt, ckpt, image, device)
        pn = tensor_to_numpy(pert)
        diff = np.abs(orig - pn)
        diff = diff / (diff.max() + 1e-8)
        ax.imshow(diff)
        ax.set_title(f'|Diff| Iter {it}', fontsize=10)
        ax.axis('off')

    # Row 3 left: Transform magnitudes
    ax_mag = fig.add_subplot(gs[2, :2])
    metrics = defaultdict(list)
    for m in all_mag:
        for k, v in m.items():
            metrics[k].append(v)
    plot_keys = [
        ('delaunay_mean', 'Delaunay', 'C0'),
        ('tps_mean', 'TPS', 'C1'),
        ('homography_norm', 'Homography', 'C2'),
        ('rs_amplitude', 'Rolling Shutter', 'C3'),
    ]
    for key, label, color in plot_keys:
        if key in metrics:
            ax_mag.plot(iters, metrics[key], lw=2, marker='o', ms=4, label=label, color=color)
    ax_mag.set_title('Transform Magnitudes', fontsize=12, fontweight='bold')
    ax_mag.set_xlabel('Iteration')
    ax_mag.set_ylabel('Magnitude')
    ax_mag.legend(fontsize=9)
    ax_mag.grid(True, alpha=0.3)

    # Row 3 right: FFT phase
    ax_fft = fig.add_subplot(gs[2, 2:])
    if 'fft_mean' in metrics:
        ax_fft.plot(iters, metrics['fft_mean'], lw=2, marker='o', ms=4, label='FFT Mean', color='C4')
        ax_fft.plot(iters, metrics['fft_max'], lw=2, marker='s', ms=4, label='FFT Max', color='C5')
    ax_fft.set_title('FFT Phase Perturbation', fontsize=12, fontweight='bold')
    ax_fft.set_xlabel('Iteration')
    ax_fft.set_ylabel('Magnitude')
    ax_fft.legend(fontsize=9)
    ax_fft.grid(True, alpha=0.3)

    # Row 4: Loss curves
    ax_loss = fig.add_subplot(gs[3, :])
    if history:
        loss_iters = [h['iteration'] for h in history]
        for k in history[0]['losses'].keys():
            vals = [h['losses'][k] for h in history]
            ax_loss.plot(loss_iters, vals, lw=2, marker='o', ms=3, label=k.replace('_', ' ').title())
    ax_loss.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend(fontsize=9, ncol=2)
    ax_loss.grid(True, alpha=0.3)

    plt.suptitle('Adversarial Geometric Perturbations - Iteration Summary',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(out_dir, 'composite_summary.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    print("  Saved: composite_summary.png")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading checkpoints...")
    checkpoints = load_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"\nFound {len(checkpoints)} checkpoints")

    # Load image
    print(f"Loading image: {args.input}")
    image = load_image(args.input, size=(224, 224), device=args.device)
    image = image.unsqueeze(0).to(args.device)

    # Create optimizer
    print("Creating optimizer...")
    optimizer, config = create_optimizer(args.device)

    # Extract data from all checkpoints
    print("Extracting transform data...")
    all_magnitudes = []
    all_iterations = []
    all_history = []
    for iteration, ckpt in checkpoints:
        all_iterations.append(iteration)
        all_magnitudes.append(extract_transform_magnitudes(ckpt))
        all_history.extend(ckpt.get('history', []))

    # Deduplicate history (checkpoints may overlap)
    seen = set()
    unique_history = []
    for h in sorted(all_history, key=lambda x: x['iteration']):
        if h['iteration'] not in seen:
            seen.add(h['iteration'])
            unique_history.append(h)
    all_history = unique_history

    # Generate visualizations
    print("\nGenerating visualizations...")

    print("  [1/5] Perturbed image grid...")
    vis_perturbed_grid(image, checkpoints, optimizer, args.output_dir, args.device, args.dpi)

    print("  [2/5] Transform magnitude curves...")
    vis_magnitude_curves(all_magnitudes, all_iterations, args.output_dir, args.dpi)

    print("  [3/5] Loss curves...")
    vis_loss_curves(all_history, args.output_dir, args.dpi)

    print("  [4/5] Per-transform details...")
    vis_per_transform(checkpoints, args.output_dir, args.dpi)

    print("  [5/5] Composite summary...")
    vis_composite(image, checkpoints, optimizer, all_magnitudes, all_iterations,
                  all_history, args.output_dir, args.device, args.dpi)

    print(f"\nAll visualizations saved to: {args.output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
