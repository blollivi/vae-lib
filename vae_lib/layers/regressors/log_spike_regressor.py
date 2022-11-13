from typing import Any

import tensorflow as tf

from .deep_regressor import DeepRegressor
from .types import LinearParams, MLPParams


class LogSpikeRegressor(DeepRegressor):
    """
        Parameters
    ----------
    linear_params : LinearParams
        Parameters of the linear layers used to compute the output.
    mlp_params : MLPParams
        Parameters of the MLP layer.
    """

    def __init__(
        self,
        linear_params: LinearParams,
        mlp_params: MLPParams = None,
        **kwargs: Any
    ) -> None:
        super().__init__(linear_params, mlp_params, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        h = super().call(inputs)
        log_spike = -tf.nn.relu(-h)  # type: ignore
        return log_spike
