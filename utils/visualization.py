"""
Visualization Utilities.

This module provides tools for visualizing adversarial perturbations
and optimization progress.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import os


class Visualizer:
    """
    Visualization tool for adversarial geometric perturbations.
    
    Provides methods for visualizing:
    - Original vs perturbed images
    - Perturbation magnitudes
    - Optimization progress
    - Landmark displacements
    
    Args:
        save_dir: Directory to save visualizations.
        figsize: Default figure size.
    """
    
    def __init__(
        self,
        save_dir: str = 'visualizations',
        figsize: Tuple[int, int] = (12, 6)
    ):
        self.save_dir = save_dir
        self.figsize = figsize
        
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_comparison(
        self,
        original: torch.Tensor,
        perturbed: torch.Tensor,
        title: str = 'Original vs Perturbed',
        save_path: str = None
    ) -> plt.Figure:
        """
        Visualize original and perturbed images side by side.
        
        Args:
            original: Original image tensor (C, H, W)
            perturbed: Perturbed image tensor (C, H, W)
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        orig_np = self._tensor_to_numpy(original)
        axes[0].imshow(orig_np)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Perturbed image
        pert_np = self._tensor_to_numpy(perturbed)
        axes[1].imshow(pert_np)
        axes[1].set_title('Perturbed')
        axes[1].axis('off')
        
        # Difference
        diff = np.abs(orig_np - pert_np)
        diff = diff / (diff.max() + 1e-8)
        axes[2].imshow(diff)
        axes[2].set_title('Difference (amplified)')
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_perturbation_magnitude(
        self,
        original: torch.Tensor,
        perturbed: torch.Tensor,
        title: str = 'Perturbation Magnitude',
        save_path: str = None
    ) -> plt.Figure:
        """
        Visualize the magnitude of perturbation.
        
        Args:
            original: Original image tensor
            perturbed: Perturbed image tensor
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original
        orig_np = self._tensor_to_numpy(original)
        axes[0].imshow(orig_np)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Perturbed
        pert_np = self._tensor_to_numpy(perturbed)
        axes[1].imshow(pert_np)
        axes[1].set_title('Perturbed')
        axes[1].axis('off')
        
        # Per-channel difference
        diff = torch.abs(original - perturbed)
        diff_magnitude = diff.mean(dim=0).cpu().numpy()
        
        im = axes[2].imshow(diff_magnitude, cmap='hot')
        axes[2].set_title('Perturbation Magnitude')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Histogram of perturbations
        diff_flat = diff.flatten().cpu().numpy()
        axes[3].hist(diff_flat, bins=50, color='blue', alpha=0.7)
        axes[3].set_title('Perturbation Distribution')
        axes[3].set_xlabel('Magnitude')
        axes[3].set_ylabel('Count')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_optimization_progress(
        self,
        history: List[Dict],
        title: str = 'Optimization Progress',
        save_path: str = None
    ) -> plt.Figure:
        """
        Visualize optimization progress over iterations.
        
        Args:
            history: List of loss dictionaries from optimization
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if not history:
            return None
        
        # Extract loss values
        iterations = [h['iteration'] for h in history]
        losses = {}
        
        for key in history[0]['losses'].keys():
            losses[key] = [h['losses'][key] for h in history]
        
        # Create figure
        n_losses = len(losses)
        fig, axes = plt.subplots(1, n_losses, figsize=(4 * n_losses, 4))
        
        if n_losses == 1:
            axes = [axes]
        
        for i, (key, values) in enumerate(losses.items()):
            axes[i].plot(iterations, values, linewidth=2)
            axes[i].set_title(key.replace('_', ' ').title())
            axes[i].set_xlabel('Iteration')
            axes[i].set_ylabel('Loss')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_landmarks(
        self,
        image: torch.Tensor,
        landmarks: torch.Tensor,
        title: str = 'Facial Landmarks',
        save_path: str = None
    ) -> plt.Figure:
        """
        Visualize facial landmarks on an image.
        
        Args:
            image: Image tensor (C, H, W)
            landmarks: Landmark coordinates (N, 2)
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # Show image
        img_np = self._tensor_to_numpy(image)
        ax.imshow(img_np)
        
        # Plot landmarks
        landmarks_np = landmarks.cpu().numpy()
        ax.scatter(landmarks_np[:, 0], landmarks_np[:, 1], c='red', s=20, alpha=0.7)
        
        # Connect landmarks for face regions
        connections = [
            (list(range(0, 17)), 'jaw'),  # Jaw line
            (list(range(17, 22)), 'left_eyebrow'),  # Left eyebrow
            (list(range(22, 27)), 'right_eyebrow'),  # Right eyebrow
            (list(range(27, 31)), 'nose_bridge'),  # Nose bridge
            (list(range(31, 36)), 'nose_tip'),  # Nose tip
            (list(range(36, 42)) + [36], 'left_eye'),  # Left eye
            (list(range(42, 48)) + [42], 'right_eye'),  # Right eye
            (list(range(48, 60)) + [48], 'outer_mouth'),  # Outer mouth
            (list(range(60, 68)) + [60], 'inner_mouth'),  # Inner mouth
        ]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(connections)))
        
        for (indices, name), color in zip(connections, colors):
            if max(indices) < len(landmarks_np):
                ax.plot(landmarks_np[indices, 0], landmarks_np[indices, 1], 
                       c=color, linewidth=1.5, alpha=0.8)
        
        ax.set_title(title)
        ax.axis('off')
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_landmark_displacement(
        self,
        original_landmarks: torch.Tensor,
        perturbed_landmarks: torch.Tensor,
        image_size: Tuple[int, int],
        title: str = 'Landmark Displacement',
        save_path: str = None
    ) -> plt.Figure:
        """
        Visualize landmark displacement between original and perturbed.
        
        Args:
            original_landmarks: Original landmark positions (N, 2)
            perturbed_landmarks: Perturbed landmark positions (N, 2)
            image_size: Image dimensions (H, W)
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        orig_np = original_landmarks.cpu().numpy()
        pert_np = perturbed_landmarks.cpu().numpy()
        
        # Plot original landmarks
        ax.scatter(orig_np[:, 0], orig_np[:, 1], c='blue', s=30, label='Original', alpha=0.7)
        
        # Plot perturbed landmarks
        ax.scatter(pert_np[:, 0], pert_np[:, 1], c='red', s=30, label='Perturbed', alpha=0.7)
        
        # Draw displacement arrows
        for i in range(len(orig_np)):
            ax.arrow(orig_np[i, 0], orig_np[i, 1],
                    pert_np[i, 0] - orig_np[i, 0],
                    pert_np[i, 1] - orig_np[i, 1],
                    head_width=3, head_length=2, fc='green', ec='green', alpha=0.5)
        
        ax.set_xlim(0, image_size[1])
        ax.set_ylim(image_size[0], 0)  # Flip y-axis for image coordinates
        ax.set_title(title)
        ax.legend()
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_grid(
        self,
        images: List[torch.Tensor],
        titles: List[str] = None,
        ncols: int = 4,
        title: str = 'Image Grid',
        save_path: str = None
    ) -> plt.Figure:
        """
        Visualize a grid of images.
        
        Args:
            images: List of image tensors
            titles: List of titles for each image
            ncols: Number of columns
            title: Overall title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        n = len(images)
        nrows = (n + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]
        
        for i, img in enumerate(images):
            row, col = i // ncols, i % ncols
            ax = axes[row][col]
            
            img_np = self._tensor_to_numpy(img)
            ax.imshow(img_np)
            
            if titles and i < len(titles):
                ax.set_title(titles[i])
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(n, nrows * ncols):
            row, col = i // ncols, i % ncols
            axes[row][col].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=150, bbox_inches='tight')
        
        return fig
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array for visualization."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        np_array = tensor.detach().cpu().numpy()
        
        # Transpose from (C, H, W) to (H, W, C)
        np_array = np.transpose(np_array, (1, 2, 0))
        
        # Clip to [0, 1]
        np_array = np.clip(np_array, 0, 1)
        
        return np_array
    
    def close_all(self):
        """Close all matplotlib figures."""
        plt.close('all')


def visualize_perturbation(
    original: torch.Tensor,
    perturbed: torch.Tensor,
    save_dir: str = 'visualizations',
    save_name: str = 'comparison.png'
) -> plt.Figure:
    """
    Convenience function to visualize perturbation.
    
    Args:
        original: Original image tensor
        perturbed: Perturbed image tensor
        save_dir: Directory to save
        save_name: Filename for saved image
        
    Returns:
        Matplotlib figure
    """
    viz = Visualizer(save_dir=save_dir)
    return viz.visualize_comparison(original, perturbed, save_path=save_name)