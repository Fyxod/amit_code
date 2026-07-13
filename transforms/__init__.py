"""
Geometric transformation modules for adversarial perturbations.
All transforms are implemented as differentiable PyTorch modules.
"""

from .fft_phase import FFTPhasePerturbation
from .delaunay import DelaunayWarp
from .homography import HomographyTransform, PiecewiseHomography
from .thin_plate_spline import ThinPlateSpline
from .rolling_shutter import RollingShutter
from .bspline_warp import BSplineWarp, create_face_mask
from .polar_warp import PolarWarp
from .bezier_warp import BezierWarp
from .lens_distortion import LensDistortion
from .mobius_warp import MobiusWarp
from .laplacian_smoothing import LaplacianSmoothingWarp
from .geodesic_warp import GeodesicWarp
from .diff_geometry_warp import DiffGeometryWarp

__all__ = [
    'FFTPhasePerturbation',
    'DelaunayWarp',
    'HomographyTransform',
    'PiecewiseHomography',
    'ThinPlateSpline',
    'RollingShutter',
    'BSplineWarp',
    'create_face_mask',
    'PolarWarp',
    'BezierWarp',
    'LensDistortion',
    'MobiusWarp',
    'LaplacianSmoothingWarp',
    'GeodesicWarp',
    'DiffGeometryWarp',
]
