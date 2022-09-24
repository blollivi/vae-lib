from typing import Any, Tuple

import numpy as np
import tensorflow as tf

from vae_lib.utils.constraints import AbsSumtoOne

from .base import BaseLayer
from .regression import DeepRegressor
from .types import DeepRegressorParams, SparseMappingParams


class SparseMapping(BaseLayer):
    """Layer that learns a sparse input to output mapping as
    a (output_dim, input_dim) matrix, called W.

    Sparsity is induced with a Spike-and-Slab Lasso prior on the weights of the
    matrix W. Maximum A Posteriori is used to evaluate W.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lambda0: float = 10,
        lambda1: float = 1,
        lambda0_step: float = 1e-2,
        a: float = 1,
        b: float = None,
    ) -> None:

        super().__init__()
        # Mask
        self.W = tf.Variable(
            initial_value=tf.random_normal_initializer()(shape=(output_dim, input_dim)),
            trainable=True,
        )
        # Probabilities that a given cell w_ij is activated.
        self.p_star = tf.Variable(
            0.5 * tf.ones((output_dim, input_dim)), trainable=False
        )
        # Expected number of features linked to a particular factor.
        self.thetas = tf.Variable(tf.random.uniform(shape=[input_dim]), trainable=False)

        self.output_dim = output_dim
        self.lambda1 = lambda1
        self.lambda0 = tf.Variable(lambda0, dtype=tf.float32, trainable=False)
        self.lambda0_step = float(lambda0_step)
        self.a = a
        self.b = b if b is not None else output_dim

    def loss(self, batch_size: int) -> float:
        w_loss = (
            self.lambda1 * self.p_star + self.lambda0 * (1 - self.p_star)
        ) * tf.abs(self.W)

        return tf.reduce_sum(w_loss) / tf.cast(batch_size, tf.float32)

    def laplace_density(self, lambda_: float, w: tf.Tensor) -> tf.Tensor:
        return tf.exp(-lambda_ * tf.abs(w))

    def map_update(self) -> None:
        "Maximum a posteriori update"
        for k in range(self.p_star.shape[1]):
            psi1 = self.laplace_density(self.lambda1, self.W[:, k])
            psi0 = self.laplace_density(self.lambda0, self.W[:, k])
            self.p_star[:, k].assign(
                self.thetas[k]
                * psi1
                / (self.thetas[k] * psi1 + (1 - self.thetas[k]) * psi0)
            )

            self.thetas[k].assign(
                (tf.reduce_sum(self.p_star[:, k]) + self.a - 1)
                / (self.a + self.b + self.output_dim - 2)  # type: ignore
            )

            self.thetas[k].assign(tf.abs(self.thetas[k]) + 1e-10)

        self.lambda0.assign(
            tf.math.minimum(float(50), self.lambda0 + self.lambda0_step)
        )

    def get_normalized_W(self) -> tf.Variable:
        return AbsSumtoOne()(tf.abs(self.W))

    def call(self, inputs: tf.Tensor) -> tf.Variable:
        self.map_update()
        self.add_loss(self.loss(batch_size=tf.shape(inputs)[0]))
        W = self.get_normalized_W()

        masked_inputs = []
        for i in range(self.output_dim):
            masked_inputs.append(tf.multiply(inputs, W[i, :]))

        return masked_inputs


class DeepSparseGaussianDistribution(BaseLayer):
    """Computes a Deep Gaussian Distribution where the output parameters
    μ and σ are computed from a masked version of the input.
    The mask is learned during the training.
    """

    def __init__(
        self,
        mu_regressor_params: DeepRegressorParams,
        logvar_regressor_params: DeepRegressorParams,
        sparse_mapping_params: SparseMappingParams,
        **kwargs: Any
    ):
        super().__init__(**kwargs)

        self.sparse_mapping = SparseMapping(**sparse_mapping_params)  # type: ignore
        self.output_dim = self.sparse_mapping.output_dim

        self.sparse_mu_regressors = [
            DeepRegressor(**mu_regressor_params) for _ in range(self.output_dim)
        ]
        self.log_var_regressor = DeepRegressor(**logvar_regressor_params)

    def compute_mu_i(
        self, Z_masked_i: tf.Tensor, i: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        mu_i = self.sparse_mu_regressors[i](Z_masked_i)

        mu_i = tf.squeeze(mu_i)

        return mu_i

    def call(self, Z: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        Z_masked_list = self.sparse_mapping(Z)

        mu_list = []

        for i, Z_masked in enumerate(Z_masked_list):
            mu_i = self.compute_mu_i(Z_masked, i)
            mu_list.append(mu_i)

        mu = tf.stack(mu_list, axis=1)
        logvar = self.log_var_regressor(Z)

        return mu, logvar

    def get_decoder_linear_weights(self) -> tf.Tensor:
        output = []
        for i in range(self.output_dim):
            output.append(
                self.deep_gaussian_distributions[i]
                .parameters_network.deep_regressors_list[0]
                .linear.weights[0]
            )
        return tf.concat(output, axis=1)

    def get_W(self) -> np.array:
        return self.sparse_mapping.get_normalized_W().numpy()
