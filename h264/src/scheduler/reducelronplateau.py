import math
from typing import Dict

import torch

from h264.src.scheduler.base_scheduler import BaseLRScheduler


class ReduceLROnPlateau(BaseLRScheduler):
    """Learning rate scheduler which decreases the learning rate if the loss
    function of interest gets stuck on a plateau, or starts to increase.
    The difference from NewBobLRScheduler is that, this one keeps a memory of
    the last step where do not observe improvement, and compares against that
    particular loss value as opposed to the most recent loss.

    Arguments
    ---------
    lr_min : float
        The minimum allowable learning rate.
    factor : float
        Factor with which to reduce the learning rate.
    patience : int
        How many epochs to wait before reducing the learning rate.

    Example
    -------
    >>> from torch.optim import Adam
    >>> from torch.nn import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=10, input_size=3)
    >>> optim = Adam(lr=1.0, params=model.parameters())
    >>> output = model(inp_tensor)
    >>> scheduler = ReduceLROnPlateau(0.25, 0.5, 2, 1)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=1, current_loss=10.0)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=2, current_loss=11.0)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=3, current_loss=13.0)
    >>> curr_lr,next_lr=scheduler([optim],current_epoch=4, current_loss=14.0)
    >>> next_lr
    0.5
    """
   
    def __init__(
        self,  
        optimizer, 
        warmup_iterations,
        warmup_init_lr,
        max_epochs = 350,
        lr_min=1e-8, 
        lr_max=0.0001,
        factor=0.9, 
        patience=2, 
        dont_halve_until_epoch=65, 
        ) -> None:

        super().__init__(warmup_iterations, warmup_init_lr)
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.factor = factor
        self.patience = patience
        self.patience_counter = 0
        self.losses = []
        self.dont_halve_until_epoch = dont_halve_until_epoch
        self.anchor = 99999
        self.max_epochs = max_epochs
        self.optimizer = optimizer

        if self.warmup_iterations > 0:
            self.warmup_step = (
                self.lr_max - self.warmup_init_lr
            ) / self.warmup_iterations

        self.period = (
            max_epochs
        )

    def get_lr(self, epoch: int, curr_iter: int) -> float:
        if curr_iter < self.warmup_iterations:
            return self.warmup_init_lr + curr_iter * self.warmup_step
        else:   
            if self.lr_multipliers is not None:
                assert len(self.lr_multipliers) == len(self.optimizer.param_groups)
                for g_id, param_group in enumerate(self.optimizer.param_groups):
                    return param_group["lr"]
            else:
                for param_group in self.optimizer.param_groups:
                    return param_group["lr"] 
            

    def update_lr(self, optimizer, epoch: int, curr_iter: int, current_loss:float):
        
        lr = self.get_lr_loss(current_epoch=epoch, curr_iter=curr_iter, current_loss=current_loss)
        lr = max(0.0, lr)
        if self.lr_multipliers is not None:
            assert len(self.lr_multipliers) == len(optimizer.param_groups)
            for g_id, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = round(
                    lr * self.lr_multipliers[g_id], self.round_places
                )
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = round(lr, self.round_places)
        return optimizer
    

    def get_lr_loss(self, current_epoch: int, curr_iter: int, current_loss:float) -> float:
        if curr_iter < self.warmup_iterations:
            return  self.warmup_init_lr + curr_iter * self.warmup_step
        elif current_loss == float("-inf"):
            return self.get_lr(epoch=current_epoch, curr_iter=curr_iter)
        
        else:
            current_lr = self.get_lr(epoch=current_epoch, curr_iter=curr_iter)

            if current_epoch <= self.dont_halve_until_epoch:
                next_lr = current_lr
                self.anchor = current_loss
            else:
                if current_loss <= self.anchor:
                    self.patience_counter = 0
                    next_lr = current_lr
                    self.anchor = current_loss
                elif (
                    current_loss > self.anchor
                    and self.patience_counter < self.patience
                ):
                    self.patience_counter = self.patience_counter + 1
                    next_lr = current_lr
                else:
                    next_lr = current_lr * self.factor
                    self.patience_counter = 0

            # impose the lower bound
            next_lr = max(next_lr, self.lr_min)

        return next_lr

