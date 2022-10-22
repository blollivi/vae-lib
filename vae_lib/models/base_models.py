from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vae_lib.layers.distribution import GaussianSampler
from vae_lib.utils.callbacks import RelativeEarlyStopping, BetaScheduler
from vae_lib.utils.losses import kl_divergence, maximum_mean_discrepancy
from .types import EarlyStoppingParams, BetaSchedulerParams



class AbstractAutoEncoder(keras.Model):
    """Modular AutoEncoder class that reads from encoder and decoder config files.
    Offers a "sklearn" type of interface.

    Parameters
    ----------
    input_dim : int
        Dimension of the input space.
    latent_dim : int
        Dimension of the latent space.
    random_state : int
        Seed of the randomness generator of tensorflow.
    encoder_params : Dict[str, Any]
        A dictionary containing the parameters required for building
        the encoder network.
    decoder_params : Dict[str, Any]
        A dictionary containing the parameters required for building
        the decoder network.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_params: Dict[str, Any],
        decoder_params: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        super().__init__()

        # Seed the model
        random_state = kwargs.get("random_state")
        if random_state is not None:
            tf.random.set_seed(random_state)

        # Save all arguments to config.
        self.config = dict(
            input_dim=input_dim,
            latent_dim=latent_dim,
            random_state=random_state,
            encoder_params=encoder_params,
            decoder_params=decoder_params,
        )

        print("Encoder Params: ", encoder_params)
        self.encoder = self._build_encoder(encoder_params)
        self.decoder = self._build_decoder(decoder_params)

    def _build_encoder(self, encoder_params: Dict) -> keras.Model:
        raise NotImplementedError

    def _build_decoder(self, decoder_params: Dict) -> keras.Model:
        raise NotImplementedError

    def call(self, X: tf.Tensor) -> tf.Tensor:
        Z = self.encoder(X)
        X_hat = self.decoder(Z)

        # Computes loss
        x_loss = self.x_loss(X, X_hat)
        loss = tf.reduce_mean(x_loss)
        self.add_loss(loss)

        return {"mu": X_hat}

    def x_loss(self, X: tf.Tensor, Xhat: tf.Tensor) -> tf.Tensor:
        """Mean Squared Error between X and Xhat"""
        return tf.reduce_mean(tf.math.squared_difference(X, Xhat), axis=1)


    def get_callbacks(self, callbacks_params: Dict[str, Dict[str, Any]]):
        early_stopping_params = callbacks_params["early_stopping"]
        early_stopping = RelativeEarlyStopping(**early_stopping_params)

        return [early_stopping]

    def fit(
        self,
        X_train: np.array,
        X_val: np.array,
        callbacks_params: Dict[str, Dict[str, Any]],
        Y_train: np.array = None,
        Y_val: np.array = None,
        **kwargs: Any
    ) -> keras.Model:
        """Fits the auto-encoder network.

        Parameters
        ----------
        X_train : np.array
            Training set.
        X_val : np.array
            Validation set.
        tol : float, optional
            Tolerance of the relative early stopping, by default 1e-3
        patience : int, optional
            Patience of the early stopping, by default 5

        Returns
        -------
        keras.Model
            The fitted model.
        """

        if Y_train is None:
            Y_train = X_train
        if Y_val is None:
            Y_val = X_val

        super().fit(
            X_train,
            Y_train,
            callbacks=self.get_callbacks(callbacks_params),
            validation_data=(X_val, Y_val),
            **kwargs
        )

        return self

    def transform(self, X: np.array) -> np.array:
        """Encodes the input array into the principal components.

        Parameters
        ----------
        X : np.array
           Input data.

        Returns
        -------
        np.array
            Principal components.
        """
        return self.encoder.predict(X, batch_size=1024)

    def inverse_transform(self, components: np.array) -> np.array:
        """Reconstructs the data from the principal components.

        Parameters
        ----------
        components : np.array
            Principal components.

        Returns
        -------
        np.array
            Reconstructed data.
        """
        return self.decoder.predict(components, batch_size=1024)

    def get_config(self) -> Dict[str, Any]:
        return self.config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> keras.Model:
        return cls(**config)


class AbstractVariationalAutoEncoder(AbstractAutoEncoder):
    """Abstract class for implementing VAE-like model.

    Parameters
    ----------
    beta : float
        Intensity of a KL regularization of the latent space vs a N(0, 1) distribution.
        By default equal to 1. If beta > 1. The model becomes a β-VAE.
    variance_type : str
        Must be of of:
            - "constant": Assumes that the variance of all features are equal and
                          constant.The reconstruction loss is then equivalent to a MSE.
            - "feature": The model learns one variance value per feature.
            - "sample": The model reconstructs a per-sample variance.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_params: Any,
        decoder_params: Any,
        beta: float = 1,
        z_loss: str = "kl",
        variance_type: str = "sample",
        **kwargs: Any
    ) -> None:
        super().__init__(
            input_dim, latent_dim, encoder_params, decoder_params, **kwargs
        )
        self.config.update(dict(beta=beta, z_loss=z_loss, variance_type=variance_type))
        self.beta = beta
        self._beta = tf.Variable(beta, trainable=False, dtype=tf.float32)

        self.gaussian_sampler = GaussianSampler()

        if variance_type != "sample":
            self.X_logvar = tf.Variable(
                initial_value=1e-6 * tf.ones(shape=(input_dim)),  # Prior std set to 1.
                trainable=(variance_type == "feature"),
            )

    def call(self, X: tf.Tensor) -> Dict[str, tf.Tensor]:
        Z_mu, Z_logvar = self.encoder(X)
        Z_sample = self.gaussian_sampler(Z_mu, Z_logvar)
        X_mu, X_logvar = self.decoder(Z_sample)

        variance_type = self.config["variance_type"]
        if variance_type != "sample":
            X_logvar = self.X_logvar

        # Computes loss
        z_loss = self.z_loss(Z_mu, Z_logvar)
        x_loss = self.x_loss(X, X_mu, X_logvar)

        loss = tf.reduce_mean(x_loss + self._beta * z_loss)
        self.add_loss(loss)

        return {"mu": X_mu, "logvar": X_logvar}

    def x_loss(  # type: ignore
        self, X: tf.Tensor, X_mu: tf.Tensor, X_logvar: tf.Tensor
    ) -> tf.Tensor:
        """Negative Log Likelihhood of X given X_mu and X_logvar"""
        X_var = tf.exp(X_logvar) + 1e-6

        log_unnormalized = -0.5 * tf.square((X - X_mu) / X_var)

        log_normalization = 0.5 * (
            tf.constant(np.log(2.0 * np.pi), dtype=tf.float32) + X_logvar
        )

        log_likelihood = tf.reduce_sum(log_unnormalized - log_normalization, axis=1)

        return -log_likelihood

    def z_loss(self, Z_mu: tf.Tensor, Z_logvar: tf.Tensor) -> tf.Tensor:

        if self.config["z_loss"] == "kl":
            # KL Divergence
            kl_divergence = -0.5 * (1 + Z_logvar - tf.square(Z_mu) - tf.exp(Z_logvar))
            z_loss = tf.reduce_sum(kl_divergence, axis=1)
        if self.config["z_loss"] == "mmd":
            # Maximum Mean Discrepancy
            z_shape = tf.shape(Z_mu)
            prior_samples = tf.random.normal(z_shape)
            Z_samples = self.gaussian_sampler(Z_mu, Z_logvar)
            z_loss = maximum_mean_discrepancy(prior_samples, Z_samples)

        return z_loss

    def transform(self, X: np.array, sample: bool = False) -> np.array:
        Z_mu, Z_logvar = self.encoder.predict(X, batch_size=1024)
        if sample:
            Z = self.gaussian_sampler(Z_mu, Z_logvar)
        else:
            Z = Z_mu
        return Z

    def inverse_transform(self, Z: np.array, sample: bool = False) -> np.array:
        X_mu, X_logvar = self.decoder.predict(Z, batch_size=1024)
        if sample:
            Xhat = self.gaussian_sampler(X_mu, X_logvar)
        else:
            Xhat = X_mu
        return Xhat


    def get_callbacks(self, callbacks_params: Dict[str, Dict[str, Any]]):
        early_stopping_params = callbacks_params["early_stopping"]
        beta_scheduler_params = callbacks_params["beta_scheduler"]

        early_stopping = RelativeEarlyStopping(**early_stopping_params)
        beta_scheduler = BetaScheduler(**beta_scheduler_params)

        return [early_stopping, beta_scheduler]