from typing import Any, Dict

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
        z_loss: str,
        variance_type: str,
        **kwargs: Any
    ) -> None:
        super().__init__(
            input_dim,
            latent_dim,
            encoder_params,
            decoder_params,
            beta,
            z_loss,
            variance_type,
            **kwargs
        )

    def _set_output_dim(
        self, params: DeepGaussianDistributionParams, output_dim: int
    ) -> DeepGaussianDistributionParams:
        params["mu_regressor_params"]["linear_params"]["output_dim"] = output_dim
        params["logvar_regressor_params"]["linear_params"]["output_dim"] = output_dim
        return params

    def _build_encoder(  # type: ignore
        self, encoder_params: Dict[str, Any]
    ) -> keras.Model:

        encoder_params = self._set_output_dim(encoder_params, self.config["latent_dim"])

        # Q(Z|X)
        self.qz_x = DeepGaussianDistribution(**encoder_params)

        X = keras.Input(shape=(self.config["input_dim"]))
        Z_mu, Z_logvar = self.qz_x(X)

        return keras.Model(inputs=X, outputs=[Z_mu, Z_logvar])

    def _build_decoder(  # type: ignore
        self, decoder_params: Dict[str, Any]
    ) -> keras.Model:

        decoder_params = self._set_output_dim(decoder_params, self.config["input_dim"])

        # P(Xhat|Z)
        self.px_z = DeepGaussianDistribution(**decoder_params)

        Z = keras.Input(shape=(self.config["latent_dim"]))
        X_mu, X_logvar = self.px_z(Z)

        return keras.Model(inputs=Z, outputs=[X_mu, X_logvar])

    def get_factors_mapping(self) -> np.array:
        weights = self.px_z.get_regressor(0).linear.weights[0].numpy()
        return weights
