from typing import Any, Tuple

import tensorflow as tf

from vae_lib.layers.base import BaseLayer
from vae_lib.layers.regressors import DeepRegressor
from vae_lib.layers.regressors.types import DeepRegressorParams



class DeepGaussianDistribution(BaseLayer):
    """Layers that computes the parameters of a multivariate diagonal gaussian
    distribution, as output of multilayer perceptron.

    Parameters
    ----------
    mean_regressor_params : DeepRegressorParams
        Parameters of the mlp and linear layers used to compute mean.
    logvar_regressor_params : DeepRegressorParams
        Parameters of the mlp and linear layers used to compute logvar.
    """

    def __init__(
        self,
        mean_regressor_params: DeepRegressorParams,
        logvar_regressor_params: DeepRegressorParams,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.mean_regressor = DeepRegressor(**mean_regressor_params)
        self.logvar_regressor = DeepRegressor(**logvar_regressor_params)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mean = self.mean_regressor(inputs)
        logvar = self.logvar_regressor(inputs)
        return mean, logvar
