"""
Utility modules for adversarial geometric perturbations.
"""

from .landmarks import LandmarkDetector, detect_landmarks
from .image_utils import load_image, save_image, preprocess_image

# Matplotlib is optional for optimization runs. Import visualization lazily so
# a minimal GPU environment can run the attack without report-only packages.
try:
    from .visualization import Visualizer, visualize_perturbation
except ModuleNotFoundError:
    Visualizer = None
    visualize_perturbation = None

__all__ = [
    'LandmarkDetector',
    'detect_landmarks',
    'Visualizer',
    'visualize_perturbation',
    'load_image',
    'save_image',
    'preprocess_image',
]
