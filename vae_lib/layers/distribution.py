from typing import Any, Tuple

import tensorflow as tf

from .base import BaseLayer
from .regression import DeepRegressor
from .types import DeepRegressorParams, LinearParams, MLPParams


class GaussianSampler(BaseLayer):
    """Layers that samples that takes parameters mu and logvar of a Gaussian
    and returns a sample.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, mu: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        eps = tf.random.normal(shape=tf.shape(mu))
        sigma = tf.exp(logvar * 0.5)
        return mu + eps * sigma


class DeepGaussianDistribution(BaseLayer):
    """Layers that computes the parameters of a multivariate diagonal gaussian
    distribution, as output of multilayer perceptron.

    Parameters
    ----------
    mu_regressor_params : DeepRegressorParams
        Parameters of the mlp and linear layers used to compute mu.
    logvar_regressor_params : DeepRegressorParams
        Parameters of the mlp and linear layers used to compute logvar.
    """

    def __init__(
        self,
        mu_regressor_params: DeepRegressorParams,
        logvar_regressor_params: DeepRegressorParams,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.mu_regressor = DeepRegressor(**mu_regressor_params)
        self.logvar_regressor = DeepRegressor(**logvar_regressor_params)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mu = self.mu_regressor(inputs)
        logvar = self.logvar_regressor(inputs)
        return mu, logvar


class SpikeSlabSampler(BaseLayer):
    def __init__(self, c: float) -> None:
        super().__init__()
        self.config["c"] = c
        self.gaussian_sampler = GaussianSampler()

    def call(self, mu: tf.Tensor, logvar: tf.Tensor, logspike: tf.Tensor) -> tf.Tensor:
        gaussian_sample = self.gaussian_sampler(mu, logvar)
        eta = tf.random.uniform(shape=tf.shape(mu))
        selection = tf.nn.sigmoid(self.config["c"] * (eta + tf.exp(logspike) - 1))
        return tf.multiply(selection, gaussian_sample)


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


class DeepSpikeSlabDistribution(BaseLayer):
    """Layers that computes the parameters of a multivariate spike and slab,
    as output of multilayer perceptron.


    Parameters
    ----------
    mu_regressor_params : DeepRegressorParams
        Parameters of the mlp and linear layers used to compute mu.
    logvar_regressor_params : DeepRegressorParams
        Parameters of the mlp and linear layers used to compute logvar.
    logspike_regressor_params: DeepRegressorParams
        Parameters of the mlp and linear layers used to compute logspike.
    """

    def __init__(
        self,
        mu_regressor_params: DeepRegressorParams,
        logvar_regressor_params: DeepRegressorParams,
        logspike_regressor_params: DeepRegressorParams,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.mu_regressor = DeepRegressor(**mu_regressor_params)
        self.logvar_regressor = DeepRegressor(**logvar_regressor_params)
        self.logspike_regressor = LogSpikeRegressor(**logspike_regressor_params)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        mu = self.mu_regressor(inputs)
        logvar = self.logvar_regressor(inputs)
        log_spike = self.logspike_regressor(inputs)
        return mu, logvar, log_spike
