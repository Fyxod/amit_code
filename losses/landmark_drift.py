"""
Landmark Drift Loss Module.

This module implements loss functions to measure and constrain facial
landmark displacement during adversarial geometric perturbations.
It ensures that key facial features (eyes, nose, mouth) remain in
consistent positions after perturbation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np


class LandmarkDriftLoss(nn.Module):
    """
    Landmark drift loss for facial landmark preservation.
    
    Measures the displacement of facial landmarks between original and
    perturbed images. The loss encourages landmarks to remain in similar
    positions after geometric perturbation.
    
    Args:
        num_landmarks: Number of facial landmarks (68 for standard face models).
        threshold: Maximum allowed landmark displacement in pixels.
        landmark_type: Type of landmarks ('face', 'body', 'custom').
        detector: Landmark detection method ('mediapipe', 'dlib', 'learnable').
        device: Device to run on.
    """
    
    def __init__(
        self,
        num_landmarks: int = 68,
        threshold: float = 5.0,
        landmark_type: str = 'face',
        detector: str = 'mediapipe',
        device: str = 'cuda'
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.threshold = threshold
        self.landmark_type = landmark_type
        self.detector = detector
        self.device = device
        
        # Initialize landmark detector
        self.landmark_detector = self._init_detector()
        
        # Define landmark groups for weighted loss
        self._init_landmark_groups()
    
    def _init_detector(self):
        """Initialize landmark detector."""
        if self.detector == 'mediapipe':
            return self._init_mediapipe()
        elif self.detector == 'learnable':
            return LearnableLandmarkDetector(num_landmarks=self.num_landmarks)
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

    
    def _init_landmark_groups(self):
        """Initialize landmark groups for weighted loss."""
        # Standard 68-point facial landmark groups
        self.landmark_groups = {
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
        
        # Weights for each group (eyes and mouth are more important)
        self.group_weights = {
            'jaw': 0.5,
            'left_eyebrow': 0.7,
            'right_eyebrow': 0.7,
            'nose_bridge': 0.8,
            'nose_tip': 0.8,
            'left_eye': 1.0,
            'right_eye': 1.0,
            'outer_mouth': 1.0,
            'inner_mouth': 1.0
        }
    
    def detect_landmarks(self, image: torch.Tensor) -> torch.Tensor:
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
        
        if self.detector == 'mediapipe' and self.landmark_detector is not None:
            landmarks = self._detect_mediapipe(image)
        elif self.detector == 'learnable':
            landmarks = self.landmark_detector(image)
        else:
            # Fallback: use center points as placeholder landmarks
            landmarks = self._get_default_landmarks(batch_size, h, w, image.device)
        
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
                result = self.landmark_detector.detect(mp_image)

                if result.face_landmarks:
                    face_lms = result.face_landmarks[0]  # list of NormalizedLandmark
                    for j, landmark in enumerate(face_lms[:self.num_landmarks]):
                        landmarks[i, j, 0] = landmark.x * w
                        landmarks[i, j, 1] = landmark.y * h

            else:
                # --- Legacy MediaPipe solutions API ---
                results = self.landmark_detector.process(img_np)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]

                    # MediaPipe face mesh has 468 landmarks, we need to sample
                    # to get 68 landmarks for compatibility
                    for j, landmark in enumerate(face_landmarks.landmark[:self.num_landmarks]):
                        landmarks[i, j, 0] = landmark.x * w
                        landmarks[i, j, 1] = landmark.y * h

        return landmarks

    
    def _get_default_landmarks(self, batch_size: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Get default landmark positions (center of image regions)."""
        landmarks = torch.zeros(batch_size, self.num_landmarks, 2, device=device)
        
        # Create a simple face landmark template
        center_x, center_y = w / 2, h / 2
        
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
    
    def forward(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor,
        original_landmarks: torch.Tensor = None,
        return_displacement: bool = False
    ) -> torch.Tensor:
        """
        Compute landmark drift loss.
        
        Args:
            original: Original image tensor of shape (B, C, H, W)
            perturbed: Perturbed image tensor of shape (B, C, H, W)
            original_landmarks: Pre-computed original landmarks (optional)
            return_displacement: If True, also return displacement values
            
        Returns:
            Loss tensor (and optionally displacement values)
        """
        # Detect landmarks
        if original_landmarks is None:
            with torch.no_grad():
                orig_landmarks = self.detect_landmarks(original)
        else:
            orig_landmarks = original_landmarks
        
        pert_landmarks = self.detect_landmarks(perturbed)
        
        # Compute displacement
        displacement = pert_landmarks - orig_landmarks
        displacement_magnitude = torch.norm(displacement, dim=2)  # (B, N)
        
        # Apply threshold (penalize only if displacement exceeds threshold)
        thresholded_displacement = F.relu(displacement_magnitude - self.threshold)
        
        # Weight by landmark groups
        weights = torch.ones(self.num_landmarks, device=self.device)
        for group_name, indices in self.landmark_groups.items():
            for idx in indices:
                if idx < self.num_landmarks:
                    weights[idx] = self.group_weights[group_name]
        
        # Weighted loss
        weighted_displacement = thresholded_displacement * weights.unsqueeze(0)
        loss = weighted_displacement.mean()
        
        if return_displacement:
            return loss, displacement_magnitude
        return loss
    
    def get_landmark_displacement(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get landmark displacement between original and perturbed images.
        
        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            
        Returns:
            Tuple of (displacement vectors, displacement magnitudes)
        """
        orig_landmarks = self.detect_landmarks(original)
        pert_landmarks = self.detect_landmarks(pertured)
        
        displacement = pert_landmarks - orig_landmarks
        magnitude = torch.norm(displacement, dim=-1)
        
        return displacement, magnitude


class LearnableLandmarkDetector(nn.Module):
    """
    Learnable landmark detection network.
    
    A simple CNN that predicts landmark positions from images.
    Can be trained on landmark datasets or fine-tuned for specific tasks.
    
    Args:
        num_landmarks: Number of landmarks to predict.
        hidden_dim: Hidden dimension of the network.
    """
    
    def __init__(
        self,
        num_landmarks: int = 68,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Landmark prediction head
        self.landmark_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_landmarks * 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict landmark positions.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Landmark coordinates of shape (B, N, 2)
        """
        features = self.features(x)
        landmarks_flat = self.landmark_head(features)
        
        # Reshape to (B, N, 2)
        batch_size = x.shape[0]
        landmarks = landmarks_flat.view(batch_size, self.num_landmarks, 2)
        
        # Scale to image dimensions
        h, w = x.shape[2], x.shape[3]
        landmarks[:, :, 0] = (landmarks[:, :, 0] + 1) * w / 2  # x coordinates
        landmarks[:, :, 1] = (landmarks[:, :, 1] + 1) * h / 2  # y coordinates
        
        return landmarks


class WeightedLandmarkLoss(nn.Module):
    """
    Weighted landmark loss with importance-based weighting.
    
    Different landmarks can have different importance weights,
    allowing focus on critical facial features.
    
    Args:
        num_landmarks: Number of landmarks.
        weights: Custom weights for each landmark.
        threshold: Maximum allowed displacement.
    """
    
    def __init__(
        self,
        num_landmarks: int = 68,
        weights: torch.Tensor = None,
        threshold: float = 5.0
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.threshold = threshold
        
        if weights is None:
            # Default weights: eyes and mouth are more important
            weights = torch.ones(num_landmarks)
            # Eyes (indices 36-47)
            weights[36:48] = 2.0
            # Mouth (indices 48-67)
            weights[48:68] = 1.5
            # Nose (indices 27-35)
            weights[27:36] = 1.2
        
        self.register_buffer('weights', weights)
    
    def forward(
        self, 
        orig_landmarks: torch.Tensor, 
        pert_landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted landmark loss.
        
        Args:
            orig_landmarks: Original landmark positions (B, N, 2)
            pert_landmarks: Perturbed landmark positions (B, N, 2)
            
        Returns:
            Weighted loss
        """
        # Compute displacement
        displacement = pert_landmarks - orig_landmarks
        magnitude = torch.norm(displacement, dim=2)  # (B, N)
        
        # Apply threshold
        thresholded = F.relu(magnitude - self.threshold)
        
        # Apply weights
        weighted = thresholded * self.weights.unsqueeze(0)
        
        return weighted.mean()


class LandmarkConsistencyLoss(nn.Module):
    """
    Landmark consistency loss for maintaining relative landmark positions.
    
    Instead of absolute positions, this loss preserves the relative
    geometry of facial landmarks.
    
    Args:
        num_landmarks: Number of landmarks.
        relative_threshold: Maximum allowed relative displacement.
    """
    
    def __init__(
        self,
        num_landmarks: int = 68,
        relative_threshold: float = 0.1
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.relative_threshold = relative_threshold
        
        # Pre-compute landmark pairs for relative distance
        self._init_landmark_pairs()
    
    def _init_landmark_pairs(self):
        """Initialize important landmark pairs for relative distance."""
        # Key facial feature pairs
        self.pairs = [
            (36, 45),  # Left eye to right eye
            (30, 51),  # Nose tip to mouth center
            (36, 48),  # Left eye to left mouth corner
            (45, 54),  # Right eye to right mouth corner
            (27, 30),  # Nose bridge to nose tip
            (48, 54),  # Mouth corners
        ]
        
        self.num_pairs = len(self.pairs)
    
    def forward(
        self, 
        orig_landmarks: torch.Tensor, 
        pert_landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute landmark consistency loss.
        
        Args:
            orig_landmarks: Original landmark positions (B, N, 2)
            pert_landmarks: Perturbed landmark positions (B, N, 2)
            
        Returns:
            Consistency loss
        """
        total_loss = 0.0
        
        for i, j in self.pairs:
            if i < self.num_landmarks and j < self.num_landmarks:
                # Original distance
                orig_dist = torch.norm(
                    orig_landmarks[:, i] - orig_landmarks[:, j], 
                    dim=1
                )
                
                # Perturbed distance
                pert_dist = torch.norm(
                    pert_landmarks[:, i] - pert_landmarks[:, j], 
                    dim=1
                )
                
                # Relative change
                relative_change = torch.abs(pert_dist - orig_dist) / (orig_dist + 1e-6)
                
                # Thresholded loss
                loss = F.relu(relative_change - self.relative_threshold).mean()
                total_loss = total_loss + loss
        
        return total_loss / self.num_pairs


class LandmarkSmoothnessLoss(nn.Module):
    """
    Smoothness constraint on landmark displacements.
    
    Encourages nearby landmarks to have similar displacements,
    preventing unrealistic local deformations.
    
    Args:
        num_landmarks: Number of landmarks.
        neighbor_weight: Weight for neighbor smoothness.
    """
    
    def __init__(
        self,
        num_landmarks: int = 68,
        neighbor_weight: float = 1.0
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.neighbor_weight = neighbor_weight
        
        # Define landmark neighbors (adjacent landmarks in face model)
        self._init_neighbors()
    
    def _init_neighbors(self):
        """Initialize neighbor relationships."""
        # For 68-point model, neighbors are sequential within each facial feature
        self.neighbors = {}
        
        # Jaw line
        for i in range(16):
            self.neighbors[i] = [i + 1]
        
        # Eyebrows
        for i in range(17, 21):
            self.neighbors[i] = [i + 1]
        for i in range(22, 26):
            self.neighbors[i] = [i + 1]
        
        # Nose
        for i in range(27, 30):
            self.neighbors[i] = [i + 1]
        
        # Eyes (circular)
        for i in range(36, 41):
            self.neighbors[i] = [i + 1]
        self.neighbors[41] = [36]
        for i in range(42, 47):
            self.neighbors[i] = [i + 1]
        self.neighbors[47] = [42]
        
        # Mouth (outer)
        for i in range(48, 59):
            self.neighbors[i] = [i + 1]
        self.neighbors[59] = [48]
        
        # Mouth (inner)
        for i in range(60, 67):
            self.neighbors[i] = [i + 1]
        self.neighbors[67] = [60]
    
    def forward(
        self, 
        orig_landmarks: torch.Tensor, 
        pert_landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute landmark smoothness loss.
        
        Args:
            orig_landmarks: Original landmark positions (B, N, 2)
            pert_landmarks: Perturbed landmark positions (B, N, 2)
            
        Returns:
            Smoothness loss
        """
        # Compute displacement
        displacement = pert_landmarks - orig_landmarks
        
        total_loss = 0.0
        count = 0
        
        for i, neighbors in self.neighbors.items():
            if i < self.num_landmarks:
                for j in neighbors:
                    if j < self.num_landmarks:
                        # Smoothness: neighboring landmarks should have similar displacement
                        diff = displacement[:, i] - displacement[:, j]
                        total_loss = total_loss + torch.norm(diff, dim=1).mean()
                        count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return total_loss * self.neighbor_weight