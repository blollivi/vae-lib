from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vae_lib.layers.distribution import DeepSpikeSlabDistribution, SpikeSlabSampler
from vae_lib.layers.types import (
    DeepGaussianDistributionParams,
    DeepSpikeSlabDistributionParams,
)

from .variational_auto_encoder import VAE


class VSC(VAE):
    """Variational Sparse Coding implementation
    https://openreview.net/pdf?id=SkeJ6iR9Km

    Parameters
    ----------
    alpha : float
        Sparsity inducing parameter. Lower values lead to sparser factors usage.
    c : float
        Parameter that controls the shape of the sigmoid function that approximates
        the step function. Higher c values lead to better approximation of the step
        function but can lead to unstable learning. (See Annex C)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_params: DeepSpikeSlabDistributionParams,
        decoder_params: DeepGaussianDistributionParams,
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
            **kwargs
        )
        self.config.update(dict(alpha=alpha, c=c))
        self.spike_and_slab_sampler = SpikeSlabSampler(c)

    def _build_encoder(  # type: ignore
        self, output_dim: int, encoder_params: DeepSpikeSlabDistributionParams
    ) -> keras.Model:
        encoder_params["output_dim"] = output_dim
        # Q(Z|X)
        self.qz_x = DeepSpikeSlabDistribution(**encoder_params)

        X = keras.Input(shape=(self.config["input_dim"]))
        Z_mu, Z_logvar, Z_logspike = self.qz_x(X)

        return keras.Model(inputs=X, outputs=[Z_mu, Z_logvar, Z_logspike])

    def z_loss(  # type: ignore
        self, Z_mu: tf.Tensor, Z_logvar: tf.Tensor, Z_logspike: tf.Tensor
    ) -> tf.Tensor:
        alpha = self.config["alpha"]

        spike = tf.clip_by_value(tf.exp(Z_logspike), 1e-6, 1.0 - 1e-6)

        kl_divergence = -0.5 * (1 + Z_logvar - tf.square(Z_mu) - tf.exp(Z_logvar))
        slab_loss = tf.multiply(spike, kl_divergence)
        spike_loss = tf.multiply(
            1 - spike, tf.math.log(1 - spike) / (1 - alpha)
        ) + tf.multiply(spike, tf.math.log(spike / alpha))

        return tf.reduce_mean(slab_loss + spike_loss, axis=1)

    def call(self, X: tf.Tensor) -> Dict[str, tf.Tensor]:
        Z_mu, Z_logvar, Z_logspike = self.encoder(X)
        Z_sample = self.spike_and_slab_sampler(Z_mu, Z_logvar, Z_logspike)
        X_mu, X_logvar = self.decoder(Z_sample)

        variance_type = self.config["variance_type"]
        if variance_type != "sample":
            X_logvar = self.X_logvar

        # Computes loss
        z_loss = self.z_loss(Z_mu, Z_logvar, Z_logspike)
        x_loss = self.x_loss(X, X_mu, X_logvar)
        loss = tf.reduce_mean(x_loss + self.config["beta"] * z_loss)
        self.add_loss(loss)

        return {"mu": X_mu, "logvar": X_logvar}

    def transform(self, X: np.array, sample: bool = False) -> np.array:
        Z_mu, Z_logvar, Z_logspike = self.encoder.predict(X)
        if sample:
            Z = self.spike_and_slab_sampler(Z_mu, Z_logvar, Z_logspike)
        else:
            Z = Z_mu
        return Z
