"""
Landmark Detection Utilities.

This module provides utilities for detecting and processing facial landmarks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict


class LandmarkDetector:
    """
    Facial landmark detector wrapper.
    
    Supports multiple backends for landmark detection.
    
    Args:
        detector_type: Type of detector ('mediapipe', 'dlib', 'learnable').
        num_landmarks: Number of landmarks to detect.
        device: Device to run on.
    """
    
    def __init__(
        self,
        detector_type: str = 'mediapipe',
        num_landmarks: int = 68,
        device: str = 'cuda'
    ):
        self.detector_type = detector_type
        self.num_landmarks = num_landmarks
        self.device = device
        
        self.detector = self._init_detector()
    
    def _init_detector(self):
        """Initialize the landmark detector."""
        if self.detector_type == 'mediapipe':
            return self._init_mediapipe()
        elif self.detector_type == 'dlib':
            try:
                import dlib
                self.predictor = dlib.shape_predictor(
                    'shape_predictor_68_face_landmarks.dat'
                )
                self.detector_dlib = dlib.get_frontal_face_detector()
                return self.predictor
            except ImportError:
                print("dlib not available")
                return None
        else:
            return None

    def _init_mediapipe(self):
        """Initialize MediaPipe landmark detector (supports both legacy and Tasks API)."""
        # --- Try the new MediaPipe Tasks API first (mediapipe >= 0.10.14) ---
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import os

            # Look for the .task model file
            model_path = os.environ.get(
                'MEDIAPIPE_FACE_LANDMARKER_MODEL', 'face_landmarker.task'
            )
            if not os.path.exists(model_path):
                print(f"MediaPipe Tasks model not found at '{model_path}', "
                      f"falling back to legacy API or default landmarks")
                return self._init_mediapipe_legacy()

            base_options = python.BaseOptions(
                model_asset_path=model_path
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._mp_api = 'tasks'
            return vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"MediaPipe Tasks API failed: {e}, trying legacy API ...")
            return self._init_mediapipe_legacy()


    def _init_mediapipe_legacy(self):
        """Initialize MediaPipe using the legacy solutions API (mediapipe < 0.10.14)."""
        try:
            import mediapipe as mp
            if not hasattr(mp, 'solutions'):
                print("MediaPipe legacy solutions API not available, "
                      "using default landmarks")
                self._mp_api = 'none'
                return None
            self.mp_face_mesh = mp.solutions.face_mesh
            self._mp_api = 'legacy'
            return self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
        except (ImportError, AttributeError) as e:
            print(f"MediaPipe legacy API not available: {e}, "
                  "using default landmarks")
            self._mp_api = 'none'
            return None

    
    def detect(self, image: torch.Tensor) -> torch.Tensor:
        """
        Detect facial landmarks in an image.
        
        Args:
            image: Input tensor of shape (B, C, H, W) or (C, H, W)
            
        Returns:
            Landmark coordinates of shape (B, N, 2) or (N, 2)
        """
        single_image = image.dim() == 3
        if single_image:
            image = image.unsqueeze(0)
        
        batch_size = image.shape[0]
        h, w = image.shape[2], image.shape[3]
        
        landmarks = torch.zeros(batch_size, self.num_landmarks, 2, device=self.device)
        
        if self.detector_type == 'mediapipe' and self.detector is not None:
            landmarks = self._detect_mediapipe(image)
        elif self.detector_type == 'dlib' and self.detector is not None:
            landmarks = self._detect_dlib(image)
        else:
            landmarks = self._get_default_landmarks(batch_size, h, w)
        
        if single_image:
            return landmarks.squeeze(0)
        return landmarks
    
    def _detect_mediapipe(self, image: torch.Tensor) -> torch.Tensor:
        """Detect landmarks using MediaPipe (Tasks or legacy API)."""
        batch_size = image.shape[0]
        h, w = image.shape[2], image.shape[3]

        landmarks = torch.zeros(batch_size, self.num_landmarks, 2, device=self.device)

        api = getattr(self, '_mp_api', 'legacy')

        for i in range(batch_size):
            # Convert to numpy
            img_np = image[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            if api == 'tasks':
                # --- New MediaPipe Tasks API ---
                import mediapipe as mp
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=img_np
                )
                result = self.detector.detect(mp_image)

                if result.face_landmarks:
                    face_lms = result.face_landmarks[0]  # list of NormalizedLandmark
                    for j, landmark in enumerate(face_lms[:self.num_landmarks]):
                        landmarks[i, j, 0] = landmark.x * w
                        landmarks[i, j, 1] = landmark.y * h

            else:
                # --- Legacy MediaPipe solutions API ---
                results = self.detector.process(img_np)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]

                    for j, landmark in enumerate(face_landmarks.landmark[:self.num_landmarks]):
                        landmarks[i, j, 0] = landmark.x * w
                        landmarks[i, j, 1] = landmark.y * h

        return landmarks

    
    def _detect_dlib(self, image: torch.Tensor) -> torch.Tensor:
        """Detect landmarks using dlib."""
        batch_size = image.shape[0]
        h, w = image.shape[2], image.shape[3]
        
        landmarks = torch.zeros(batch_size, self.num_landmarks, 2, device=self.device)
        
        for i in range(batch_size):
            # Convert to numpy
            img_np = image[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_gray = np.mean(img_np, axis=2).astype(np.uint8)
            
            faces = self.detector_dlib(img_gray)
            
            if faces:
                shape = self.predictor(img_gray, faces[0])
                
                for j in range(min(self.num_landmarks, 68)):
                    landmarks[i, j, 0] = shape.part(j).x
                    landmarks[i, j, 1] = shape.part(j).y
        
        return landmarks
    
    def _get_default_landmarks(
        self, 
        batch_size: int, 
        h: int, 
        w: int
    ) -> torch.Tensor:
        """Get default landmark positions."""
        landmarks = torch.zeros(batch_size, self.num_landmarks, 2, device=self.device)
        
        center_x, center_y = w / 2, h / 2
        
        # Create a simple face template
        # Jaw line
        for i in range(17):
            landmarks[:, i, 0] = center_x - 50 + i * 6.25
            landmarks[:, i, 1] = center_y + 30 + abs(i - 8) * 2
        
        # Eyebrows
        for i in range(5):
            landmarks[:, 17 + i, 0] = center_x - 40 + i * 10
            landmarks[:, 17 + i, 1] = center_y - 30
            landmarks[:, 22 + i, 0] = center_x + 10 + i * 10
            landmarks[:, 22 + i, 1] = center_y - 30
        
        # Nose
        for i in range(9):
            landmarks[:, 27 + i, 0] = center_x + (i % 3 - 1) * 10
            landmarks[:, 27 + i, 1] = center_y - 10 + (i // 3) * 10
        
        # Eyes
        for i in range(6):
            angle = i * np.pi / 3
            landmarks[:, 36 + i, 0] = center_x - 30 + 10 * np.cos(angle)
            landmarks[:, 36 + i, 1] = center_y - 15 + 5 * np.sin(angle)
            landmarks[:, 42 + i, 0] = center_x + 20 + 10 * np.cos(angle)
            landmarks[:, 42 + i, 1] = center_y - 15 + 5 * np.sin(angle)
        
        # Mouth
        for i in range(12):
            angle = i * np.pi / 6
            landmarks[:, 48 + i, 0] = center_x + 20 * np.cos(angle)
            landmarks[:, 48 + i, 1] = center_y + 25 + 10 * np.sin(angle)
        
        return landmarks


def detect_landmarks(
    image: torch.Tensor,
    detector_type: str = 'mediapipe',
    num_landmarks: int = 68
) -> torch.Tensor:
    """
    Convenience function to detect landmarks.
    
    Args:
        image: Input image tensor
        detector_type: Type of detector
        num_landmarks: Number of landmarks
        
    Returns:
        Landmark coordinates
    """
    detector = LandmarkDetector(detector_type, num_landmarks, str(image.device))
    return detector.detect(image)


def draw_landmarks(
    image: torch.Tensor,
    landmarks: torch.Tensor,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> torch.Tensor:
    """
    Draw landmarks on an image.
    
    Args:
        image: Input image tensor (C, H, W)
        landmarks: Landmark coordinates (N, 2)
        color: RGB color for drawing
        
    Returns:
        Image with landmarks drawn
    """
    image = image.clone()
    
    for i in range(landmarks.shape[0]):
        x, y = int(landmarks[i, 0]), int(landmarks[i, 1])
        
        if 0 <= x < image.shape[2] and 0 <= y < image.shape[1]:
            # Draw a small circle
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx * dx + dy * dy <= 4:
                        px, py = x + dx, y + dy
                        if 0 <= px < image.shape[2] and 0 <= py < image.shape[1]:
                            image[:, py, px] = torch.tensor(color, dtype=image.dtype) / 255.0
    
    return image


def get_landmark_groups(num_landmarks: int = 68) -> Dict[str, List[int]]:
    """
    Get landmark group indices for standard 68-point model.
    
    Args:
        num_landmarks: Number of landmarks
        
    Returns:
        Dictionary mapping group names to landmark indices
    """
    if num_landmarks == 68:
        return {
            'jaw': list(range(0, 17)),
            'left_eyebrow': list(range(17, 22)),
            'right_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_tip': list(range(31, 36)),
            'left_eye': list(range(36, 42)),
            'right_eye': list(range(42, 48)),
            'outer_mouth': list(range(48, 60)),
            'inner_mouth': list(range(60, 68))
        }
    else:
        return {'all': list(range(num_landmarks))}


def compute_landmark_distances(
    landmarks1: torch.Tensor,
    landmarks2: torch.Tensor
) -> torch.Tensor:
    """
    Compute distances between corresponding landmarks.
    
    Args:
        landmarks1: First set of landmarks (N, 2)
        landmarks2: Second set of landmarks (N, 2)
        
    Returns:
        Distance for each landmark (N,)
    """
    return torch.norm(landmarks1 - landmarks2, dim=1)


def align_landmarks(
    landmarks: torch.Tensor,
    reference: torch.Tensor
) -> torch.Tensor:
    """
    Align landmarks to a reference using Procrustes analysis.
    
    Args:
        landmarks: Landmarks to align (N, 2)
        reference: Reference landmarks (N, 2)
        
    Returns:
        Aligned landmarks
    """
    # Center
    landmarks_centered = landmarks - landmarks.mean(dim=0)
    reference_centered = reference - reference.mean(dim=0)
    
    # Scale
    landmarks_scale = torch.norm(landmarks_centered)
    reference_scale = torch.norm(reference_centered)
    
    landmarks_scaled = landmarks_centered / (landmarks_scale + 1e-8)
    reference_scaled = reference_centered / (reference_scale + 1e-8)
    
    # Rotation (using SVD)
    H = landmarks_scaled.T @ reference_scaled
    U, S, V = torch.svd(H)
    R = V @ U.T
    
    # Apply transformation
    aligned = (landmarks_scaled @ R) * reference_scale + reference.mean(dim=0)
    
    return aligned