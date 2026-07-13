# Adversarial Geometric Perturbations

A PyTorch library for generating adversarial geometric perturbations with full optimization loop support. This library implements differentiable geometric transformations that can be optimized to create adversarial examples while preserving perceptual quality.

## Features

### Geometric Transformations
- **FFT Phase Perturbation**: Frequency domain phase manipulation
- **Delaunay Triangulation Warp**: Triangle-based mesh deformation
- **Homography Transformation**: 8-DOF perspective transformation
- **Thin-Plate Spline (TPS) Warp**: Non-rigid smooth deformation
- **Rolling Shutter Effect**: Camera sensor artifact simulation

### Loss Functions
- **Adversarial Loss**: Targeted/untargeted attacks on face recognition models
- **Identity Drift Loss**: Preserve identity features during perturbation
- **Landmark Drift Loss**: Maintain facial landmark positions
- **LPIPS Loss**: Perceptual similarity constraint
- **Regularization**: Smoothness and total variation constraints

### Target Models
- FaceNet
- ArcFace
- Custom models

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

```bash
# Basic usage with all transforms
python main.py -i path/to/image.jpg -o output.png --visualize

# Use specific transforms
python main.py -i image.jpg --fft --tps

# Custom settings
python main.py -i image.jpg --iterations 1000 --lr 0.02 --visualize
```

### Python API

```python
import torch
from config import Config
from optimization import AdversarialOptimizer
from models import FaceRecognitionModel
from utils import load_image, save_image

# Load configuration
config = Config()

# Load image
image = load_image('image.jpg', size=(224, 224))
image = image.unsqueeze(0).to(config.device)

# Create target model
target_model = FaceRecognitionModel(device=config.device)

# Create optimizer
optimizer = AdversarialOptimizer(config, target_model, config.device)

# Run optimization
perturbed, results = optimizer.optimize(image)

# Save result
save_image(perturbed[0], 'perturbed.png')
```

### Using Individual Transforms

```python
from transforms import FFTPhasePerturbation, ThinPlateSpline

# FFT Phase Perturbation
fft = FFTPhasePerturbation(image_size=(224, 224), magnitude=0.5)
perturbed = fft(image)

# Thin-Plate Spline Warp
tps = ThinPlateSpline(image_size=(224, 224), max_displacement=15.0)
warped = tps(image)
```

## Project Structure

```
adv_geometric_perturbations/
├── config.py              # Configuration
├── main.py                # Main entry point
├── requirements.txt       # Dependencies
├── transforms/            # Geometric transformations
│   ├── fft_phase.py
│   ├── delaunay.py
│   ├── homography.py
│   ├── thin_plate_spline.py
│   └── rolling_shutter.py
├── losses/                # Loss functions
│   ├── identity_drift.py
│   ├── landmark_drift.py
│   ├── lpips_loss.py
│   ├── adversarial_loss.py
│   └── combined_loss.py
├── optimization/          # Optimization modules
│   ├── optimizer.py
│   ├── perturbation_params.py
│   └── schedulers.py
├── models/                # Target model wrappers
│   └── target_model.py
├── utils/                 # Utility functions
│   ├── landmarks.py
│   ├── visualization.py
│   └── image_utils.py
└── examples/              # Example scripts
    └── example_usage.py
```

## Configuration

Key configuration parameters:

```python
# Transform settings
fft_enabled = True
fft_magnitude = 0.5
delaunay_enabled = True
delaunay_max_displacement = 10.0
homography_enabled = True
homography_max_perturbation = 0.1
tps_enabled = True
tps_max_displacement = 15.0
rolling_shutter_enabled = True
rolling_shutter_max_offset = 10.0

# Loss weights
adversarial_weight = 1.0
identity_weight = 0.5
landmark_weight = 0.3
lpips_weight = 0.2

# Optimization
num_iterations = 500
learning_rate = 0.01
optimizer = 'adam'
```

## Examples

Run the example script:

```bash
python examples/example_usage.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{adv_geometric_perturbations,
  title = {Adversarial Geometric Perturbations},
  year = {2024},
  description = {A PyTorch library for generating adversarial geometric perturbations}
}
```

## License

MIT License