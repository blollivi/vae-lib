from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vae_lib.layers.distributions.types import DeepGaussianDistributionParams

from .variational_auto_encoder import VAE


def hard_probs(x):
    return tf.clip_by_value(x, 0.0 + 1e-6, 1.0 - 1e-6)
    

def log_normal_diag(x, mean, logvar, reduce_dim=None, name=None):
    """
    Multivariate log normal
    @param reduce_dim: dimension of the data attributes, along which to sum the log-prob.
        If tensor has shape (minibatch, sample_size) then provide reduce_dim=1
        If tensor has shape (N, L, sample_size) then provide reduce_dim=2
    """
    log2pi = np.log(2 * np.pi)
    log_normal = -.5 * (log2pi + logvar + tf.math.pow(x - mean, 2) / tf.math.exp(logvar))
    return tf.reduce_sum(log_normal, axis=reduce_dim, name=name)


class VampPriorVAE(VAE):
    """Vamp Prior VAE
    https://arxiv.org/pdf/1705.07120.pdf
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_params: DeepGaussianDistributionParams,
        decoder_params: DeepGaussianDistributionParams,
        beta: float,
        variance_type: str,
        n_components: int,
        l1_reg: float=0,
        pseudoinputs_mean: float = 0.5,
        pseudoinputs_std: float = 0.01,
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

        self.config.update(dict(n_components=n_components))
        self.n_components = n_components
        self.C = tf.math.log(tf.constant(self.n_components, dtype=tf.float32))

        # trainable
        self.means = keras.layers.Dense(
            units=input_dim,
            use_bias=False,
            kernel_initializer=tf.initializers.RandomNormal(
                mean=pseudoinputs_mean, stddev=pseudoinputs_std
            ),
            kernel_regularizer=tf.keras.regularizers.l1(l1_reg)
        )
        self.means.build((None, n_components))

        # create an idle input for calling pseudo-inputs
        self.idle_input = tf.eye(
            num_rows=n_components,
            num_columns=n_components,
            dtype=tf.float32
        )


    def z_loss(  # type: ignore
        self, Z_mean: tf.Tensor, Z_logvar: tf.Tensor, Z_sample: tf.Tensor
    ) -> tf.Tensor:

        # Loss due to prior regularization
        # Prior: Vamp Prior
        # 1. get mean and var from pseudo_inputs
        pseudo_inputs = self.means(self.idle_input)
        pseudo_mean, pseudo_logvar = self.encoder(pseudo_inputs)  # C x K
        Z_sample_expand = tf.expand_dims(Z_sample, 1)  # N x 1 x K
        pseudo_mean_expand = tf.expand_dims(pseudo_mean, 0)  # 1 x C x K
        pseudo_logvar_expand = tf.expand_dims(pseudo_logvar, 0)  # 1 x C x K

        lognormal = log_normal_diag(
            Z_sample_expand, pseudo_mean_expand, pseudo_logvar_expand,
            reduce_dim=2, name='pseudo-log-normal'
        ) - self.C  # N x C

        log_p = tf.reduce_logsumexp(lognormal, axis=-1)  # N

        # 2. Posterior: Normal posterior
        log_q = log_normal_diag(
            Z_sample, Z_mean, Z_logvar, reduce_dim=1
        )

        return log_q - log_p
