from typing import Any, List, Tuple

import tensorflow as tf

from .base import BaseLayer, Identity
from .features_extraction import Linear, MultiLayerPerceptron
from .types import DeepRegressorParams, LinearParams, MLPParams


class DeepRegressor(BaseLayer):
    """Model Y = (Y₁, ..., Yn) = f(X) avec f=Sequential([MLP, linear layer])

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
        super().__init__(**kwargs)

        if mlp_params is None or len(mlp_params) == 0:
            self.mlp = Identity()
        else:
            self.mlp = MultiLayerPerceptron(**mlp_params)

        self.linear = Linear(**linear_params)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.mlp is not None:
            h = self.mlp(inputs)
        else:
            h = inputs
        return self.linear(h)


class DeepParallelRegressors(BaseLayer):
    """Model Y₁=f₁(X), ..., Yn=fn(X), where each fᵢ is a deep regressor.
    The number of output variables: n, is refered as the number of chanels.
    """

    def __init__(
        self, deep_regressor_params_list: List[DeepRegressorParams], **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.n_chanels = len(deep_regressor_params_list)

        self.deep_regressors_list = [
            DeepRegressor(**deep_regressor_params)
            for deep_regressor_params in deep_regressor_params_list
        ]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs_list = []
        for deep_regressor in self.deep_regressors_list:
            outputs_list.append(deep_regressor(inputs))
        return tf.stack(outputs_list, axis=0)
