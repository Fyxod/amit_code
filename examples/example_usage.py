#!/usr/bin/env python
"""
Example usage of Adversarial Geometric Perturbations.

This script demonstrates how to use the library programmatically.
"""

import torch
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, update_config
from optimization import AdversarialOptimizer
from models import FaceRecognitionModel
from transforms import (
    FFTPhasePerturbation,
    DelaunayWarp,
    HomographyTransform,
    ThinPlateSpline,
    RollingShutter
)
from losses import CombinedLoss
from utils import load_image, save_image, Visualizer


def example_basic_usage():
    """Basic usage example with default settings."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Configuration
    config = Config()
    device = config.device
    
    # Create a sample image (random for demonstration)
    image = torch.rand(1, 3, 224, 224).to(device)
    
    # Create target model
    target_model = FaceRecognitionModel(device=device)
    
    # Create optimizer
    optimizer = AdversarialOptimizer(config, target_model, device)
    
    # Run optimization
    perturbed, results = optimizer.optimize(image, num_iterations=100)
    
    print(f"Original image shape: {image.shape}")
    print(f"Perturbed image shape: {perturbed.shape}")
    print(f"Best loss: {results['best_loss']:.6f}")
    print()


def example_individual_transforms():
    """Example using individual transforms."""
    print("=" * 60)
    print("Example 2: Individual Transforms")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = torch.rand(1, 3, 224, 224).to(device)
    
    # FFT Phase Perturbation
    print("\n1. FFT Phase Perturbation:")
    fft = FFTPhasePerturbation(image_size=(224, 224), magnitude=0.5)
    fft = fft.to(device)
    fft_out = fft(image)
    print(f"   Output shape: {fft_out.shape}")
    print(f"   Perturbation magnitude: {fft.get_perturbation_magnitude():.6f}")
    
    # Delaunay Triangulation
    print("\n2. Delaunay Triangulation Warp:")
    delaunay = DelaunayWarp(image_size=(224, 224), num_points=16, max_displacement=10.0)
    delaunay = delaunay.to(device)
    delaunay_out = delaunay(image)
    print(f"   Output shape: {delaunay_out.shape}")
    print(f"   Displacement magnitude: {delaunay.get_displacement_magnitude():.6f}")
    
    # Homography
    print("\n3. Homography Transform:")
    homography = HomographyTransform(image_size=(224, 224), max_perturbation=0.1)
    homography = homography.to(device)
    homography_out = homography(image)
    print(f"   Output shape: {homography_out.shape}")
    print(f"   Perturbation magnitude: {homography.get_perturbation_magnitude():.6f}")
    
    # Thin-Plate Spline
    print("\n4. Thin-Plate Spline Warp:")
    tps = ThinPlateSpline(image_size=(224, 224), num_control_points=16, max_displacement=15.0)
    tps = tps.to(device)
    tps_out = tps(image)
    print(f"   Output shape: {tps_out.shape}")
    print(f"   Displacement magnitude: {tps.get_displacement_magnitude():.6f}")
    
    # Rolling Shutter
    print("\n5. Rolling Shutter Effect:")
    rs = RollingShutter(image_size=(224, 224), max_offset=10.0)
    rs = rs.to(device)
    rs_out = rs(image)
    print(f"   Output shape: {rs_out.shape}")
    print(f"   Offset magnitude: {rs.get_offset_magnitude():.6f}")
    
    print()


def example_custom_config():
    """Example with custom configuration."""
    print("=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)
    
    # Create custom config
    config = Config()
    config = update_config(config,
        # Enable only specific transforms
        fft_enabled=True,
        tps_enabled=True,
        homography_enabled=False,
        delaunay_enabled=False,
        rolling_shutter_enabled=False,
        
        # Custom loss weights
        adversarial_weight=1.0,
        identity_weight=0.3,
        landmark_weight=0.2,
        lpips_weight=0.1,
        
        # Optimization settings
        num_iterations=200,
        learning_rate=0.02,
        
        # Transform parameters
        fft_magnitude=0.3,
        tps_max_displacement=10.0
    )
    
    device = config.device
    image = torch.rand(1, 3, 224, 224).to(device)
    
    # Create optimizer with custom config
    target_model = FaceRecognitionModel(device=device)
    optimizer = AdversarialOptimizer(config, target_model, device)
    
    # Run optimization
    perturbed, results = optimizer.optimize(image)
    
    print(f"Configuration:")
    print(f"  FFT enabled: {config.fft_enabled}")
    print(f"  TPS enabled: {config.tps_enabled}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"Best loss: {results['best_loss']:.6f}")
    print()


def example_loss_functions():
    """Example demonstrating loss functions."""
    print("=" * 60)
    print("Example 4: Loss Functions")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create sample images
    original = torch.rand(1, 3, 224, 224).to(device)
    perturbed = original + torch.randn_like(original) * 0.1
    perturbed = torch.clamp(perturbed, 0, 1)
    
    # Create config
    config = Config()
    config.device = device
    
    # Create combined loss
    target_model = FaceRecognitionModel(device=device)
    loss_fn = CombinedLoss(config, target_model, device)
    
    # Compute losses
    total_loss, losses = loss_fn(original, perturbed, return_components=True)
    
    print("Loss values:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.6f}")
    print()


def example_with_real_image():
    """Example with a real image (if available)."""
    print("=" * 60)
    print("Example 5: With Real Image")
    print("=" * 60)
    
    # Check for sample image
    sample_path = "sample_image.jpg"
    
    if not os.path.exists(sample_path):
        print(f"No sample image found at {sample_path}")
        print("Skipping this example.")
        print()
        return
    
    # Load image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = load_image(sample_path, size=(224, 224), device=device)
    image = image.unsqueeze(0)
    
    print(f"Loaded image shape: {image.shape}")
    
    # Create config
    config = Config()
    config.num_iterations = 100
    config.device = device
    
    # Create optimizer
    target_model = FaceRecognitionModel(device=device)
    optimizer = AdversarialOptimizer(config, target_model, device)
    
    # Run optimization
    perturbed, results = optimizer.optimize(image)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    save_image(perturbed[0], 'results/perturbed.png')
    
    print(f"Perturbed image saved to results/perturbed.png")
    print(f"Best loss: {results['best_loss']:.6f}")
    print()


def example_visualization():
    """Example showing visualization capabilities."""
    print("=" * 60)
    print("Example 6: Visualization")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create sample images
    original = torch.rand(3, 224, 224).to(device)
    perturbed = original + torch.randn_like(original) * 0.05
    perturbed = torch.clamp(perturbed, 0, 1)
    
    # Create visualizer
    viz = Visualizer(save_dir='results/visualizations')
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Comparison
    fig1 = viz.visualize_comparison(
        original, perturbed,
        title='Original vs Perturbed',
        save_path='comparison.png'
    )
    
    # Perturbation magnitude
    fig2 = viz.visualize_perturbation_magnitude(
        original, perturbed,
        title='Perturbation Magnitude',
        save_path='magnitude.png'
    )
    
    # Optimization history
    history = [
        {'iteration': i, 'losses': {'total': 1.0 - i * 0.01, 'adversarial': 0.5 - i * 0.005}}
        for i in range(100)
    ]
    fig3 = viz.visualize_optimization_progress(
        history,
        title='Optimization Progress',
        save_path='progress.png'
    )
    
    viz.close_all()
    
    print("Visualizations saved to results/visualizations/")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ADVERSARIAL GEOMETRIC PERTURBATIONS - EXAMPLES")
    print("=" * 60 + "\n")
    
    # Run examples
    example_basic_usage()
    example_individual_transforms()
    example_custom_config()
    example_loss_functions()
    example_with_real_image()
    example_visualization()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()