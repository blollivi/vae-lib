from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vae_lib.layers.sparsity import DeepSparseGaussianDistribution
from vae_lib.layers.types import (
    DeepGaussianDistributionParams,
    DeepSparseGaussianDistributionParams,
    DeepSpikeSlabDistributionParams
)

from .variational_sparse_coding import VSC


class SparseVSC(VSC):
    """ Mix of VSC and SparseVAE"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_params: DeepSpikeSlabDistributionParams,
        decoder_params: DeepSparseGaussianDistributionParams,
        beta: float,
        variance_type: str,
        alpha: float,
        c: float,
        **kwargs: Any
    ) -> None:

        super().__init__(
            input_dim,
            latent_dim,
            encoder_params,
            decoder_params,
            beta,
            variance_type,
            alpha,
            c,
            **kwargs
        )

        if "sigma_prior" in kwargs:
            self.X_logvar = tf.Variable(
                2
                * tf.math.log(
                    tf.convert_to_tensor(
                        kwargs.get["sigma_prior"],  # type: ignore
                        dtype=tf.float32,
                    )
                )
            )

        self.sigma_prior_df = kwargs.get("sigma_prior_df", 3)
        self.sigma_prior_scale = kwargs.get("sigma_prior_scale", 1)

    def _build_decoder(  # type: ignore
        self, decoder_params: DeepSparseGaussianDistributionParams
    ) -> keras.Model:
        decoder_params["sparse_mapping_params"]["output_dim"] = self.config["input_dim"]
        decoder_params["sparse_mapping_params"]["input_dim"] = self.config["latent_dim"]
        decoder_params["mu_regressor_params"]["linear_params"]["output_dim"] = 1
        decoder_params["logvar_regressor_params"]["linear_params"][
            "output_dim"
        ] = self.config["input_dim"]

        # P(Xhat|Z)
        self.px_z = DeepSparseGaussianDistribution(**decoder_params)

        Z = keras.Input(shape=(self.config["latent_dim"]))
        X_mu, X_logvar = self.px_z(Z)

        return keras.Model(inputs=Z, outputs=[X_mu, X_logvar])

    def sigma_loss(self, batch_size: int) -> float:
        sigma_loss = (batch_size + self.sigma_prior_df + 2) * tf.reduce_sum(
            0.5 * self.X_logvar
        ) + 0.5 * self.sigma_prior_df * self.sigma_prior_scale * tf.reduce_sum(
            1 / tf.exp(self.X_logvar)
        )

        sigma_loss = sigma_loss / batch_size
        return sigma_loss

    def x_loss(  # type: ignore
        self, X: tf.Tensor, X_mu: tf.Tensor, X_logvar: tf.Tensor
    ) -> tf.Tensor:
        """Negative Log Likelihhood of X given X_mu and X_logvar"""
        X_var = tf.exp(X_logvar)

        log_unnormalized = -0.5 * tf.square(X - X_mu) / X_var

        if self.config["variance_type"] != "feature":
            log_normalization = 0.5 * (
                tf.constant(np.log(2.0 * np.pi), dtype=tf.float32) + X_logvar
            )
        else:
            batch_size = tf.cast(tf.shape(X)[0], tf.float32)
            log_normalization = self.sigma_loss(batch_size)
            self.add_metric(log_normalization, "sigma_loss")

        log_likelihood = tf.reduce_sum(log_unnormalized - log_normalization, axis=1)

        return -log_likelihood

    def get_factors_mapping(self) -> np.array:
        mask = self.px_z.get_W().T
        decoder_linear_weights = self.px_z.get_decoder_linear_weights().numpy()

        return mask * decoder_linear_weights
