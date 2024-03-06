import math
from typing import Dict

from h264.src.scheduler.base_scheduler import BaseLRScheduler

class CosineScheduler(BaseLRScheduler):
    """
    Cosine learning rate scheduler: https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        warmup_iterations,
        warmup_init_lr,
        is_iteration_based=True,
        max_iterations=150000,
        max_epochs=350,
        min_lr=1e-5,
        max_lr=0.4,
    ) -> None:
        super().__init__(warmup_iterations, warmup_init_lr)

        is_iter_based = is_iteration_based
        max_iterations = max_iterations
        self.min_lr = min_lr
        self.max_lr = max_lr

        if self.warmup_iterations > 0:
            self.warmup_step = (
                self.max_lr - self.warmup_init_lr
            ) / self.warmup_iterations

        self.period = (
            max_iterations - self.warmup_iterations + 1 if is_iter_based else max_epochs
        )

        self.is_iter_based = is_iter_based

    def get_lr(self, epoch: int, curr_iter: int) -> float:
        if curr_iter < self.warmup_iterations:
            curr_lr = self.warmup_init_lr + curr_iter * self.warmup_step
            self.warmup_epochs = epoch
        else:
            if self.is_iter_based:
                curr_iter = curr_iter - self.warmup_iterations
                curr_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                    1 + math.cos(math.pi * curr_iter / self.period)
                )
            else:
                adjust_num = self.warmup_epochs + 1 if self.adjust_period else 0
                adjust_den = self.warmup_epochs if self.adjust_period else 0
                curr_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                    1
                    + math.cos(
                        math.pi * (epoch - adjust_num) / (self.period - adjust_den)
                    )
                )
        return max(0.0, curr_lr)
