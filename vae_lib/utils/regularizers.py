from typing import Any

import tensorflow as tf


class L1(tf.keras.regularizers.L1):
    def __init__(self, l1: float, **kwargs: Any) -> None:
        super().__init__(l1, **kwargs)

    def __call__(self, x: tf.Tensor) -> float:
        return self.l1 * tf.reduce_mean(tf.abs(x))
