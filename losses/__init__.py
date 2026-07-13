"""
Loss functions for adversarial geometric perturbations.
"""

from .identity_drift import IdentityDriftLoss
from .landmark_drift import LandmarkDriftLoss
from .lpips_loss import LPIPSLoss
from .adversarial_loss import AdversarialLoss
from .combined_loss import CombinedLoss
from .embedding_loss import TAESDEmbeddingLoss, MultiScaleEmbeddingLoss, LatentConsistencyLoss

__all__ = [
    'IdentityDriftLoss',
    'LandmarkDriftLoss',
    'LPIPSLoss',
    'AdversarialLoss',
    'CombinedLoss',
    'TAESDEmbeddingLoss',
    'MultiScaleEmbeddingLoss',
    'LatentConsistencyLoss',
]
