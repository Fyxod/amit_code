"""
Configuration file for adversarial geometric perturbations.
All hyperparameters and settings can be modified here.
"""

import torch

class Config:
    """Main configuration class for adversarial geometric perturbations."""
    
    # Image settings
    image_size = (512, 512)
    batch_size = 1
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Optimization settings
    num_iterations = 500
    learning_rate = 0.01
    optimizer = 'adam'  # 'adam', 'sgd', 'lbfgs'
    
    # Learning rate scheduling
    use_scheduler = True
    scheduler_type = 'step'  # 'step', 'cosine', 'exponential'
    step_size = 100
    gamma = 0.5
    
    # Early stopping
    early_stopping = True
    patience = 50
    min_delta = 1e-6
    
    # Checkpointing
    save_checkpoints = True
    checkpoint_dir = 'checkpoints'
    checkpoint_interval = 50
    
    # ==================== Transform Parameters ====================
    
    # FFT Phase Perturbation
    fft_enabled = True
    fft_magnitude = 0.1  # Max phase perturbation magnitude (reduced from 0.5)
    fft_learnable = True
    fft_phase_resolution = (28, 28)  # Coarse grid resolution (None = full resolution)
    fft_shared_channels = True  # Share phase across channels (3x param reduction)
    
    # Delaunay Triangulation
    delaunay_enabled = True
    delaunay_num_points = 64  # Number of control points (increased from 16)
    delaunay_max_displacement = 3.0  # Max pixel displacement (reduced from 10.0)
    delaunay_learnable = True
    
    # Homography
    homography_enabled = True
    homography_max_perturbation = 0.05  # Max homography parameter value (reduced from 0.1)
    homography_learnable = True
    homography_piecewise = True  # Use piecewise homography instead of global
    homography_grid_size = (4, 4)  # Grid size for piecewise homography
    
    # Thin-Plate Spline
    tps_enabled = True
    tps_num_control_points = 64  # Grid of control points (increased from 16)
    tps_max_displacement = 5.0  # Max pixel displacement (reduced from 15.0)
    tps_learnable = True
    
    # Rolling Shutter
    rolling_shutter_enabled = True
    rolling_shutter_max_offset = 3.0  # Max pixel offset (reduced from 10.0)
    rolling_shutter_direction = 'horizontal'  # 'horizontal' or 'vertical'
    rolling_shutter_learnable = True
    rolling_shutter_num_harmonics = 8  # Number of harmonic modes (increased from 2)
    rolling_shutter_per_row_amplitude = True  # Per-row amplitude for spatial variation
    
    # ==================== Loss Weights ====================
    
    # Adversarial loss (maximize target model confusion)
    adversarial_weight = 1.0
    adversarial_target = 'untargeted'  # 'untargeted' or 'targeted'
    target_class = None  # For targeted attacks
    
    # Identity drift loss (preserve identity features)
    identity_weight = 0.5
    identity_threshold = 0.3  # Maximum allowed identity drift
    
    # Landmark drift loss (preserve facial landmarks)
    landmark_weight = 0.3
    landmark_threshold = 5.0  # Maximum allowed landmark displacement (pixels)
    
    # LPIPS perceptual loss
    lpips_weight = 1.0  # Enabled by default to constrain visible distortion
    lpips_net = 'alex'  # 'alex', 'vgg', 'squeeze'
    lpips_threshold = 0.3  # Maximum allowed perceptual distance
    
    # Embedding loss (TAESD)
    embedding_weight = 0.2
    embedding_loss_type = 'l2'  # 'l1', 'l2', 'cosine'
    taesd_path = 'taesd'  # Path to TAESD model
    
    # Smoothness regularization
    smoothness_weight = 0.1
    
    # Total variation regularization
    tv_weight = 0.01
    
    # ==================== Target Model Settings ====================
    
    target_model = 'facenet'  # 'facenet', 'arcface', 'custom'
    target_model_path = None  # Path to custom model weights
    
    # ==================== Visualization Settings ====================
    
    visualize = True
    visualization_interval = 20
    save_visualizations = True
    visualization_dir = 'visualizations'
    
    # ==================== Logging Settings ====================
    
    log_interval = 10
    log_dir = 'logs'
    verbose = True


class TransformConfig:
    """Configuration for individual transforms."""
    
    class FFTPhase:
        enabled = True
        magnitude = 0.1  # Reduced from 0.5
        frequency_range = (0.1, 0.9)  # Low and high frequency cutoff
        learnable = True
        phase_resolution = (28, 28)  # Coarse grid resolution (None = full)
        shared_channels = True  # Share phase across channels
        
    class Delaunay:
        enabled = True
        num_points = 64  # Increased from 16
        max_displacement = 3.0  # Reduced from 10.0
        border_mode = 'constant'  # 'constant', 'reflect', 'replicate'
        learnable = True
        
    class Homography:
        enabled = True
        max_perturbation = 0.05  # Reduced from 0.1
        preserve_aspect_ratio = False
        learnable = True
        piecewise = True  # Use piecewise homography
        grid_size = (4, 4)  # Grid for piecewise homography
        
    class ThinPlateSpline:
        enabled = True
        num_control_points = 64  # Increased from 16
        max_displacement = 5.0  # Reduced from 15.0
        regularization = 0.0
        learnable = True
        
    class RollingShutter:
        enabled = True
        max_offset = 3.0  # Reduced from 10.0
        direction = 'horizontal'
        wave_frequency = 1.0
        learnable = True
        num_harmonics = 8  # Increased from 2
        per_row_amplitude = True  # Per-row spatial variation


class LossConfig:
    """Configuration for loss functions."""
    
    class AdversarialLoss:
        weight = 1.0
        target = 'untargeted'
        target_class = None
        confidence_threshold = 0.5
        
    class IdentityDrift:
        weight = 0.5
        threshold = 0.3
        feature_layer = 'avgpool'  # Layer to extract features from
        
    class LandmarkDrift:
        weight = 0.3
        threshold = 5.0
        landmark_type = 'face'  # 'face', 'body', 'custom'
        num_landmarks = 68
        
    class LPIPS:
        weight = 1.0  # Enabled by default to constrain visible distortion
        net = 'alex'
        threshold = 0.3
        spatial = False  # Use spatial LPIPS
        
    class Smoothness:
        weight = 0.1
        order = 2  # 1 for gradient, 2 for Laplacian
        
    class TotalVariation:
        weight = 0.01


def get_config():
    """Get the default configuration."""
    return Config()


def update_config(config, **kwargs):
    """Update configuration with keyword arguments."""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    return config


def print_config(config):
    """Print current configuration."""
    print("=" * 50)
    print("Configuration")
    print("=" * 50)
    for attr in dir(config):
        if not attr.startswith('_') and not callable(getattr(config, attr)):
            print(f"{attr}: {getattr(config, attr)}")
    print("=" * 50)