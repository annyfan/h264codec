class BaseLRScheduler(object):
    def __init__(
        self,
        warmup_iterations=0,
        warmup_epochs=0,
        warmup_init_lr=1e-7,
        adjust_period_for_epochs=False,
        lr_multipliers=None,
    ) -> None:
        super().__init__()
        self.round_places = 8
        self.lr_multipliers = lr_multipliers

        warmup_iterations = warmup_iterations
        self.warmup_iterations = max(warmup_iterations, 0)

        self.warmup_init_lr = warmup_init_lr

        # Because of variable batch sizes, we can't determine exact number of epochs in warm-up phase. This
        # may result in different LR schedules when we run epoch- and iteration-based schedulers.
        # To reduce these differences, we use adjust_period_for_epochs arguments.
        # For epoch-based scheduler, this parameter value should be enabled.
        self.adjust_period = adjust_period_for_epochs
        self.warmup_epochs = warmup_epochs

    def get_lr(self, epoch: int, curr_iter: int):
        raise NotImplementedError

    def update_lr(self, optimizer, epoch: int, curr_iter: int):
        lr = self.get_lr(epoch=epoch, curr_iter=curr_iter)
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

    @staticmethod
    def retrieve_lr(optimizer) -> list:
        lr_list = []
        for param_group in optimizer.param_groups:
            lr_list.append(param_group["lr"])
        return lr_list