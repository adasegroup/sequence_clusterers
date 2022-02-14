from pytorch_lightning.callbacks import Callback
from typing import Optional


class DecayLearningRate(Callback):
    def __init__(self,
                 lr_update_param: float,
                 lr_update_tol: int,
                 min_lr: Optional[float],
                 updated_lr: Optional[float]):
        self.old_lrs = []
        self.lr_update_param = lr_update_param
        self.lr_update_tol = lr_update_tol
        self.min_lr = min_lr
        self.updated_lr = updated_lr
        self.checker = 0

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = [param_group["lr"] for param_group in optimizer.param_groups]
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.cur_loss >= pl_module.prev_loss:
            self.checker += 1
        if self.checker >= self.lr_update_tol:
            self.checker = 0
            for opt_idx, optimizer in enumerate(trainer.optimizers):
                old_lr_group = self.old_lrs[opt_idx]
                new_lr_group = []
                for p_idx, param_group in enumerate(optimizer.param_groups):
                    old_lr = old_lr_group[p_idx]
                    new_lr = old_lr * self.lr_update_param
                    if self.min_lr is not None:
                        if new_lr < self.min_lr:
                            new_lr = self.updated_lr
                    new_lr_group.append(new_lr)
                    param_group["lr"] = new_lr
                self.old_lrs[opt_idx] = new_lr_group
