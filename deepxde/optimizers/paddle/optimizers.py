__all__ = ["get", "is_external_optimizer"]

import paddle


def is_external_optimizer(optimizer):
    return optimizer in ["L-BFGS", "L-BFGS-B"]


def get(params, optimizer, learning_rate=None, decay=None):
    """Retrieves an Optimizer instance."""
    if isinstance(optimizer, paddle.optimizer.Optimizer):
        return optimizer

    if optimizer in ["L-BFGS", "L-BFGS-B"]:
        # TODO: add support for L-BFGS and L-BFGS-B
        raise ValueError("L-BFGS to be implemented for backend Paddle.")

    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    if decay is not None:
        # TODO: learning rate decay
        learning_rate = _get_lr_scheduler(learning_rate, decay)
    if optimizer == "adam":
        return paddle.optimizer.Adam(learning_rate=learning_rate, parameters=params)
    raise NotImplementedError(f"{optimizer} to be implemented for backend Paddle.")


class InverseTimeDecay(paddle.optimizer.lr.InverseTimeDecay):
    def __init__(self, learning_rate, gamma, decay_steps=1, last_epoch=-1, verbose=False):
        self.decay_steps = decay_steps
        super(InverseTimeDecay, self).__init__(learning_rate, gamma, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr / (1 + self.gamma * (self.last_epoch / self.decay_steps))


def _get_lr_scheduler(lr, decay):
    if decay is None:
        return lr, None
    if decay[0] == "inverse time":
        lr_sch = InverseTimeDecay(
            lr,  # 初始学习率
            decay[2],  # 衰减系数
            decay[1],  # 每隔decay[1]步衰减
            verbose=False
        )
    else:
        raise NotImplementedError(
            f"{decay[0]} decay to be implemented for backend tensorflow.compat.v1."
        )
    return lr_sch
