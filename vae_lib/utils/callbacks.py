from typing import Any

import numpy as np
from tensorflow import keras


class RelativeEarlyStopping(keras.callbacks.EarlyStopping):
    """Early Stopping that monitors relative improvement.

    Parameters
    ----------
    tol : float
        Tolerance for the stopping condition.
    """

    def __init__(self, tol: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tol = tol
        self.epoch = 1

    def _is_improvement(self, monitor_value: float, reference_value: float) -> bool:
        if reference_value == np.Inf:
            reference_value = 1000
        delta = reference_value - monitor_value
        return self.monitor_op(self.tol * monitor_value, delta)
