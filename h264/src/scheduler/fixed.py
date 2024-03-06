import math
from typing import Dict

from h264.src.scheduler.base_scheduler import BaseLRScheduler


class FixedLRScheduler(BaseLRScheduler):
    """
    Fixed learning rate scheduler with optional linear warm-up strategy
    """

    def __init__(     
        self,   
        warmup_iterations,
        warmup_init_lr,
        is_iteration_based=True,
        max_iterations=150000,
        max_epochs=350,
        lr=0.0001) -> None:
        is_iter_based = is_iteration_based
        super().__init__(warmup_iterations, warmup_init_lr)

        self.max_iterations = max_iterations

        self.fixed_lr = lr

        if self.warmup_iterations > 0:
            self.warmup_step = (
                self.fixed_lr - self.warmup_init_lr
            ) / self.warmup_iterations

        self.period = (
            max_iterations - self.warmup_iterations + 1
            if is_iter_based
            else max_epochs
        )

        self.is_iter_based = is_iter_based


    def get_lr(self, epoch: int, curr_iter: int) -> float:
        if curr_iter < self.warmup_iterations:
            curr_lr = self.warmup_init_lr + curr_iter * self.warmup_step
        else:
            curr_lr = self.fixed_lr
        return max(0.0, curr_lr)