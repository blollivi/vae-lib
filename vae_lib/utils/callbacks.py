from typing import Any

import numpy as np
from tensorflow import keras


class RelativeEarlyStopping(keras.callbacks.EarlyStopping):
    """Early Stopping that monitors relative improvement.
     A prediction is considered as an improvement is its represent
    an improvement of at least an improvement of <tol> times the 
    previous best prediction.

    Parameters
    ----------
    tol : float
        Tolerance for the stopping condition.
    """

    def __init__(
        self, 
        tol: float = 0.01,
        monitor: str = "val_loss",
        patience: int = 5,
        restore_best_weights: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            monitor=monitor,
            patience=patience,
            restore_best_weights=restore_best_weights,
             **kwargs
        )
        self.tol = tol
        self.epoch = 1

    def _is_improvement(self, monitor_value: float, reference_value: float) -> bool:
        if reference_value == np.Inf:
            reference_value = 1000
        delta = reference_value - monitor_value
        return self.monitor_op(self.tol * monitor_value, delta)



class BetaScheduler(keras.callbacks.Callback):

    def __init__(self, warmup_epochs: int = 0, delay_epochs: int = 0) -> None:
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.delay_epochs = delay_epochs
        self.is_active = warmup_epochs > 0

    def on_epoch_end(self, epoch: int, logs=None):
        if self.is_active:
            initial_beta =  self.model.beta
            slope = initial_beta / self.warmup_epochs

            weight = self._compute_weight(epoch, slope)
            keras.backend.set_value(self.model._beta, float(weight * initial_beta))

        print(" - Beta : ", self.model._beta.numpy())

    def _compute_weight(self, epoch, slope):
        if epoch <= self.delay_epochs:
            return 0
        if epoch > self.delay_epochs:
            return min(
                self.model.beta,
                slope * (epoch - self.delay_epochs)
            )
             
