from typing import Any, Tuple

import tensorflow as tf

from vae_lib.layers.base import BaseLayer
from vae_lib.layers.regressors import DeepRegressor, LogSpikeRegressor
from vae_lib.layers.regressors.types import DeepRegressorParams


class DeepSpikeSlabDistribution(BaseLayer):
    """Layers that computes the parameters of a multivariate spike and slab,
    as output of multilayer perceptron.


    Parameters
    ----------
    mean_regressor_params : DeepRegressorParams
        Parameters of the mlp and linear layers used to compute mean.
    logvar_regressor_params : DeepRegressorParams
        Parameters of the mlp and linear layers used to compute logvar.
    logspike_regressor_params: DeepRegressorParams
        Parameters of the mlp and linear layers used to compute logspike.
    """

    def __init__(
        self,
        mean_regressor_params: DeepRegressorParams,
        logvar_regressor_params: DeepRegressorParams,
        logspike_regressor_params: DeepRegressorParams,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.mean_regressor = DeepRegressor(**mean_regressor_params)
        self.logvar_regressor = DeepRegressor(**logvar_regressor_params)
        self.logspike_regressor = LogSpikeRegressor(**logspike_regressor_params)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        mean = self.mean_regressor(inputs)
        logvar = self.logvar_regressor(inputs)
        log_spike = self.logspike_regressor(inputs)
        return mean, logvar, log_spike
