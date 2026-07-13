#!/usr/bin/env python
"""
Main script for Adversarial Geometric Perturbations.

This script provides a command-line interface for generating adversarial
geometric perturbations using various transformation types.
"""

import argparse
import torch
import os
import sys

from config import Config, update_config, print_config
from optimization import AdversarialOptimizer
from models import FaceRecognitionModel, create_target_model
from utils import load_image, save_image, Visualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Adversarial Geometric Perturbations'
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', '-o', type=str, default='output.png',
                        help='Path to save perturbed image')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory for output files')
    
    # Model settings
    parser.add_argument('--target-model', type=str, default='facenet',
                        choices=['facenet', 'arcface', 'resnet50', 'resnet18'],
                        help='Target model to attack')
    parser.add_argument('--model-type', type=str, default='face',
                        choices=['face', 'classification'],
                        help='Type of target model')
    
    # Attack settings
    parser.add_argument('--attack-type', type=str, default='untargeted',
                        choices=['untargeted', 'targeted', 'dodging'],
                        help='Type of adversarial attack')
    parser.add_argument('--target-class', type=int, default=None,
                        help='Target class for targeted attack')
    
    # Optimization settings
    parser.add_argument('--iterations', type=int, default=500,
                        help='Number of optimization iterations')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer type')
    
    # Transform settings
    parser.add_argument('--fft', action='store_true',
                        help='Enable FFT phase perturbation')
    parser.add_argument('--delaunay', action='store_true',
                        help='Enable Delaunay triangulation warp')
    parser.add_argument('--homography', action='store_true',
                        help='Enable homography transformation')
    parser.add_argument('--tps', action='store_true',
                        help='Enable thin-plate spline warp')
    parser.add_argument('--rolling-shutter', action='store_true',
                        help='Enable rolling shutter effect')
    parser.add_argument('--all-transforms', action='store_true',
                        help='Enable all transformations')
    
    # FFT phase parameters
    parser.add_argument('--fft-phase-resolution', type=int, nargs=2, default=None,
                        help='Coarse grid resolution for FFT phase (e.g., 28 28). '
                             'None = full resolution (original behavior)')
    parser.add_argument('--fft-shared-channels', action='store_true',
                        help='Share phase parameters across channels (3x reduction)')
    parser.add_argument('--fft-full-resolution', action='store_true',
                        help='Use full resolution FFT phase (disables coarse grid)')
    parser.add_argument('--fft-separate-channels', action='store_true',
                        help='Use separate phase parameters per channel (disables sharing)')
    
    # Delaunay parameters
    parser.add_argument('--delaunay-num-points', type=int, default=None,
                        help='Number of Delaunay control points (default: 64)')
    
    # Homography parameters
    parser.add_argument('--piecewise-homography', action='store_true',
                        help='Use piecewise homography instead of global')
    parser.add_argument('--global-homography', action='store_true',
                        help='Use global homography (disable piecewise)')
    parser.add_argument('--homography-grid-size', type=int, nargs=2, default=None,
                        help='Grid size for piecewise homography (e.g., 4 4)')
    
    # TPS parameters
    parser.add_argument('--tps-num-control-points', type=int, default=None,
                        help='Number of TPS control points (default: 64)')
    
    # Rolling shutter parameters
    parser.add_argument('--rs-num-harmonics', type=int, default=None,
                        help='Number of rolling shutter harmonic modes (default: 8)')
    parser.add_argument('--rs-per-row-amplitude', action='store_true',
                        help='Use per-row amplitude for rolling shutter')
    parser.add_argument('--rs-scalar-amplitude', action='store_true',
                        help='Use scalar amplitude for rolling shutter (disable per-row)')
    
    # Loss weights
    parser.add_argument('--adv-weight', type=float, default=0,
                        help='Adversarial loss weight')
    parser.add_argument('--identity-weight', type=float, default=0,
                        help='Identity drift loss weight')
    parser.add_argument('--landmark-weight', type=float, default=0,
                        help='Landmark drift loss weight')
    parser.add_argument('--lpips-weight', type=float, default=1.0,
                        help='LPIPS perceptual loss weight (default: 1.0)')
    parser.add_argument('--embedding-weight', type=float, default=0.2,
                        help='Embedding loss weight (TAESD)')
    parser.add_argument('--embedding-loss-type', type=str, default='l2',
                        choices=['l1', 'l2', 'cosine'],
                        help='Embedding loss distance metric')
    parser.add_argument('--taesd-path', type=str, default='taesd',
                        help='Path to TAESD model')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--vis-interval', type=int, default=50,
                        help='Visualization interval')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Save perturbed image every N iterations')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save per-iteration images (default: <output-dir>/iterations)')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create configuration
    config = Config()
    
    # Update config from arguments
    config = update_config(config,
        # Optimization
        num_iterations=args.iterations,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        device=args.device,
        
        # Attack settings
        adversarial_target=args.attack_type,
        target_class=args.target_class,
        target_model=args.target_model,
        
        # Loss weights
        adversarial_weight=args.adv_weight,
        identity_weight=args.identity_weight,
        landmark_weight=args.landmark_weight,
        lpips_weight=args.lpips_weight,
        embedding_weight=args.embedding_weight,
        embedding_loss_type=args.embedding_loss_type,
        taesd_path=args.taesd_path,
        
        # Visualization
        visualize=args.visualize,
        visualization_interval=args.vis_interval,
        verbose=args.verbose
    )
    
    # Enable transforms
    if args.all_transforms:
        config.fft_enabled = True
        config.delaunay_enabled = True
        config.homography_enabled = True
        config.tps_enabled = True
        config.rolling_shutter_enabled = True
    else:
        config.fft_enabled = args.fft
        config.delaunay_enabled = args.delaunay
        config.homography_enabled = args.homography
        config.tps_enabled = args.tps
        config.rolling_shutter_enabled = args.rolling_shutter
    
    # If no transform specified, enable all
    if not any([config.fft_enabled, config.delaunay_enabled, 
                config.homography_enabled, config.tps_enabled,
                config.rolling_shutter_enabled]):
        config.fft_enabled = True
        config.delaunay_enabled = True
        config.homography_enabled = True
        config.tps_enabled = True
        config.rolling_shutter_enabled = True
    
    # Apply FFT phase parameters
    if args.fft_full_resolution:
        config.fft_phase_resolution = None
    elif args.fft_phase_resolution is not None:
        config.fft_phase_resolution = tuple(args.fft_phase_resolution)
    
    if args.fft_separate_channels:
        config.fft_shared_channels = False
    elif args.fft_shared_channels:
        config.fft_shared_channels = True
    
    # Apply Delaunay parameters
    if args.delaunay_num_points is not None:
        config.delaunay_num_points = args.delaunay_num_points
    
    # Apply Homography parameters
    if args.global_homography:
        config.homography_piecewise = False
    elif args.piecewise_homography:
        config.homography_piecewise = True
    
    if args.homography_grid_size is not None:
        config.homography_grid_size = tuple(args.homography_grid_size)
    
    # Apply TPS parameters
    if args.tps_num_control_points is not None:
        config.tps_num_control_points = args.tps_num_control_points
    
    # Apply Rolling Shutter parameters
    if args.rs_num_harmonics is not None:
        config.rolling_shutter_num_harmonics = args.rs_num_harmonics
    
    if args.rs_scalar_amplitude:
        config.rolling_shutter_per_row_amplitude = False
    elif args.rs_per_row_amplitude:
        config.rolling_shutter_per_row_amplitude = True
    
    if args.verbose:
        print_config(config)
    
    # Load image
    print(f"Loading image: {args.input}")
    image = load_image(args.input, size=config.image_size, device=config.device)
    image = image.unsqueeze(0)  # Add batch dimension
    
    print(f"Image shape: {image.shape}")
    
    # Create target model
    print(f"Loading target model: {args.target_model}")
    if args.model_type == 'face':
        target_model = FaceRecognitionModel(
            model_name=args.target_model,
            device=config.device
        )
    else:
        target_model = create_target_model(
            model_type=args.model_type,
            model_name=args.target_model,
            device=config.device
        )
    
    # Set up per-iteration save directory
    iter_save_dir = args.save_dir if args.save_dir else os.path.join(args.output_dir, 'iterations')
    os.makedirs(iter_save_dir, exist_ok=True)
    
    # Create optimizer
    print("Initializing optimizer...")
    optimizer = AdversarialOptimizer(config, target_model, config.device)
    
    # Run optimization
    print(f"Running optimization for {config.num_iterations} iterations...")
    print(f"Saving perturbed images every {args.save_interval} iteration(s) to: {iter_save_dir}")
    
    def callback(iteration, losses, perturbed):
        """Callback for saving perturbed images and visualizations during optimization."""
        if iteration % args.save_interval == 0:
            # Detach perturbed from computation graph for saving/visualization
            perturbed_detached = perturbed.detach()
            
            # Save perturbed image
            img_path = os.path.join(iter_save_dir, f'perturbed_{iteration:04d}.png')
            save_image(perturbed_detached[0], img_path)
            
            # Save perturbation visualization
            viz = Visualizer(save_dir=iter_save_dir)
            viz.visualize_comparison(
                image[0], perturbed_detached[0],
                title=f'Iteration {iteration}',
                save_path=f'vis_{iteration:04d}.png'
            )
            viz.visualize_perturbation_magnitude(
                image[0], perturbed_detached[0],
                title=f'Perturbation Magnitude - Iter {iteration}',
                save_path=f'pert_mag_{iteration:04d}.png'
            )
            viz.close_all()
            
            if args.verbose:
                loss_str = ', '.join(f'{k}: {v.item():.4f}' for k, v in losses.items())
                print(f"  Iter {iteration}: saved image + viz | {loss_str}")
    
    perturbed, results = optimizer.optimize(
        image,
        num_iterations=config.num_iterations,
        callback=callback
    )
    
    # Save results
    output_path = os.path.join(args.output_dir, args.output)
    print(f"Saving perturbed image to: {output_path}")
    save_image(perturbed[0], output_path)
    
    # Generate final visualizations
    if args.visualize:
        print("Generating visualizations...")
        viz = Visualizer(save_dir=args.output_dir)
        
        # Comparison
        viz.visualize_comparison(
            image[0], perturbed[0],
            title='Original vs Perturbed',
            save_path='final_comparison.png'
        )
        
        # Perturbation magnitude
        viz.visualize_perturbation_magnitude(
            image[0], perturbed[0],
            title='Perturbation Magnitude',
            save_path='perturbation_magnitude.png'
        )
        
        # Optimization progress
        if results.get('history'):
            viz.visualize_optimization_progress(
                results['history'],
                title='Optimization Progress',
                save_path='optimization_progress.png'
            )
        
        viz.close_all()
    
    # Print results
    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)
    print(f"Best loss: {results['best_loss']:.6f}")
    print(f"Final loss: {results['final_loss']:.6f}")
    print(f"Iterations: {results['num_iterations']}")
    
    # Print transform magnitudes
    magnitudes = optimizer.get_transform_magnitudes()
    print("\nTransform magnitudes:")
    for name, mag in magnitudes.items():
        print(f"  {name}: {mag:.6f}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()