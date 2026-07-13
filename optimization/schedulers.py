"""
Learning Rate Schedulers Module.

This module provides learning rate schedulers for the optimization process.
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List, Optional


class WarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup.
    
    Gradually increases learning rate during warmup period, then applies
    the base scheduler.
    
    Args:
        optimizer: Optimizer to schedule.
        warmup_epochs: Number of warmup epochs.
        warmup_lr: Starting learning rate for warmup.
        after_scheduler: Scheduler to use after warmup.
        last_epoch: Last epoch index.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        warmup_lr: float = 1e-6,
        after_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_lr + alpha * (base_lr - self.warmup_lr)
                for base_lr in self.base_lrs
            ]
        
        if self.after_scheduler:
            if not self.finished_warmup:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished_warmup = True
            return self.after_scheduler.get_last_lr()
        
        return self.base_lrs
    
    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        
        if epoch >= self.warmup_epochs and self.after_scheduler:
            if not self.finished_warmup:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished_warmup = True
            
            self.after_scheduler.step(epoch - self.warmup_epochs)
        
        super().step(epoch)


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine annealing with warm restarts.
    
    Args:
        optimizer: Optimizer to schedule.
        T_0: Period of the first restart.
        T_mult: Multiplier for period increase after restart.
        eta_min: Minimum learning rate.
        last_epoch: Last epoch index.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        return [
            self.eta_min + (base_lr - self.eta_min) * 
            (1 + math.cos(math.pi * self.T_cur / self.T_0)) / 2
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        
        if epoch >= self.T_0:
            self.T_cur = epoch - self.T_0
            self.T_0 = self.T_0 * self.T_mult
        else:
            self.T_cur = epoch
        
        super().step(epoch)


class CyclicLR(_LRScheduler):
    """
    Cyclic learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule.
        base_lr: Minimum learning rate.
        max_lr: Maximum learning rate.
        step_size: Number of iterations per half cycle.
        mode: 'triangular', 'triangular2', or 'exp_range'.
        gamma: Scaling factor for 'exp_range' mode.
        last_epoch: Last epoch index.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        max_lr: float,
        step_size: int,
        mode: str = 'triangular',
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale = 1.0
        elif self.mode == 'triangular2':
            scale = 1.0 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale = self.gamma ** self.last_epoch
        else:
            scale = 1.0
        
        return [
            self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale
            for _ in self.base_lrs
        ]


class OneCycleLR(_LRScheduler):
    """
    One Cycle learning rate scheduler.
    
    Implements the 1cycle policy: https://arxiv.org/abs/1708.07120
    
    Args:
        optimizer: Optimizer to schedule.
        max_lr: Maximum learning rate.
        total_steps: Total number of steps.
        pct_start: Percentage of steps for increasing LR.
        anneal_strategy: 'cos' or 'linear'.
        div_factor: Initial LR = max_lr / div_factor.
        final_div_factor: Final LR = max_lr / final_div_factor.
        last_epoch: Last epoch index.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        super().__init__(optimizer, last_epoch)
    
    def _get_anneal_func(self, start: float, end: float, pct: float) -> float:
        if self.anneal_strategy == 'cos':
            return end + (start - end) * (1 + math.cos(math.pi * pct)) / 2
        else:  # linear
            return start + (end - start) * pct
    
    def get_lr(self) -> List[float]:
        pct = self.last_epoch / self.total_steps
        
        if pct < self.pct_start:
            # Increasing phase
            lr = self._get_anneal_func(
                self.initial_lr, self.max_lr, pct / self.pct_start
            )
        else:
            # Decreasing phase
            lr = self._get_anneal_func(
                self.max_lr, self.final_lr,
                (pct - self.pct_start) / (1 - self.pct_start)
            )
        
        return [lr for _ in self.base_lrs]


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    **kwargs
) -> _LRScheduler:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule.
        scheduler_type: Type of scheduler.
        **kwargs: Additional arguments for the scheduler.
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 100),
            gamma=kwargs.get('gamma', 0.5)
        )
    
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 500),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.99)
        )
    
    elif scheduler_type == 'cyclic':
        return CyclicLR(
            optimizer,
            base_lr=kwargs.get('base_lr', 1e-6),
            max_lr=kwargs.get('max_lr', 1e-3),
            step_size=kwargs.get('step_size', 100),
            mode=kwargs.get('mode', 'triangular')
        )
    
    elif scheduler_type == 'one_cycle':
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            total_steps=kwargs.get('total_steps', 500),
            pct_start=kwargs.get('pct_start', 0.3)
        )
    
    elif scheduler_type == 'warmup_cosine':
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 500) - kwargs.get('warmup_epochs', 50),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
        return WarmupScheduler(
            optimizer,
            warmup_epochs=kwargs.get('warmup_epochs', 50),
            warmup_lr=kwargs.get('warmup_lr', 1e-6),
            after_scheduler=base_scheduler
        )
    
    elif scheduler_type == 'cosine_restarts':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 100),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    else:
        # Default: no scheduler
        return None