from typing import Any, Dict

import tensorflow as tf
from tensorflow import keras


class BaseLayer(keras.layers.Layer):
    """Abstract base layer class.
    Parameters
    ----------
    output_dim : int
        Output dimension.
    """

    def __init__(self, output_dim: int = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.config: Dict[str, Any] = dict(output_dim=output_dim)

    def update_config(self, new_config: Dict[str, Any]) -> None:
        self.config.update(new_config)

    def get_config(self) -> Dict[str, Any]:
        return self.config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> keras.Model:
        return cls(**config)


class Identity(BaseLayer):
    """Identity function"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs
