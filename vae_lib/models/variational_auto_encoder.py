from typing import Any

import numpy as np
from tensorflow import keras

from vae_lib.layers.distribution import DeepGaussianDistribution
from vae_lib.layers.types import DeepGaussianDistributionParams

from .base_models import AbstractVariationalAutoEncoder


class VAE(AbstractVariationalAutoEncoder):
    """Implements a VAE where encoder and decoder are of dense layers only."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_params: DeepGaussianDistributionParams,
        decoder_params: DeepGaussianDistributionParams,
        beta: float,
        variance_type: str,
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

    def _build_encoder(  # type: ignore
        self, output_dim: int, encoder_params: DeepGaussianDistributionParams
    ) -> keras.Model:
        encoder_params["output_dim"] = output_dim
        # Q(Z|X)
        self.qz_x = DeepGaussianDistribution(**encoder_params)

        X = keras.Input(shape=(self.config["input_dim"]))
        Z_mu, Z_logvar = self.qz_x(X)

        return keras.Model(inputs=X, outputs=[Z_mu, Z_logvar])

    def _build_decoder(  # type: ignore
        self, output_dim: int, decoder_params: DeepGaussianDistributionParams
    ) -> keras.Model:

        decoder_params["output_dim"] = output_dim
        # Q(Z|X)
        # P(Xhat|Z)
        self.px_z = DeepGaussianDistribution(**decoder_params)

        Z = keras.Input(shape=(self.config["latent_dim"]))
        X_mu, X_logvar = self.px_z(Z)

        return keras.Model(inputs=Z, outputs=[X_mu, X_logvar])

    def get_factors_mapping(self) -> np.array:
        weights = self.px_z.get_regressor(0).linear.weights[0].numpy()
        return weights
