from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vae_lib.layers.distribution import GaussianSampler
from vae_lib.utils.callbacks import RelativeEarlyStopping


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
        self.config: Dict[str, Any] = dict(
            input_dim=input_dim,
            latent_dim=latent_dim,
            random_state=random_state,
            encoder_params=encoder_params,
            decoder_params=decoder_params,
        )

        self.encoder = self._build_encoder(encoder_params, output_dim=latent_dim)
        self.decoder = self._build_decoder(decoder_params, output_dim=input_dim)

    def _build_encoder(self, encoder_params: Dict, output_dim: int) -> keras.Model:
        raise NotImplementedError

    def _build_decoder(self, decoder_params: Dict, output_dim: int) -> keras.Model:
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

    def fit(
        self,
        X_train: np.array,
        X_val: np.array,
        tol: float = 1e-3,
        patience: int = 5,
        monitor: str = "val_loss",
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
        earlystop = RelativeEarlyStopping(
            monitor=monitor,
            tol=tol,
            patience=patience,
            restore_best_weights=True,
        )
        if Y_train is None:
            Y_train = X_train
        if Y_val is None:
            Y_val = X_val

        super().fit(
            X_train,
            Y_train,
            callbacks=[earlystop],
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
        return self.encoder.predict(X)

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
        return self.decoder.predict(components)

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
        variance_type: str = "sample",
        **kwargs: Any
    ) -> None:
        super().__init__(
            input_dim, latent_dim, encoder_params, decoder_params, **kwargs
        )
        self.config.update(dict(beta=beta, variance_type=variance_type))

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
        loss = tf.reduce_mean(x_loss + self.config["beta"] * z_loss)
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
        """KL Divergence between N(Z_mu, exp(Z_logvar)) and N(0, 0)"""
        kl_divergence = -0.5 * (1 + Z_logvar - tf.square(Z_mu) - tf.exp(Z_logvar))

        return tf.reduce_sum(kl_divergence, axis=1)

    def transform(self, X: np.array, sample: bool = False) -> np.array:
        Z_mu, Z_logvar = self.encoder.predict(X)
        if sample:
            Z = self.gaussian_sampler(Z_mu, Z_logvar)
        else:
            Z = Z_mu
        return Z

    def inverse_transform(self, Z: np.array, sample: bool = False) -> np.array:
        X_mu, X_logvar = self.decoder.predict(Z)
        if sample:
            Xhat = self.gaussian_sampler(X_mu, X_logvar)
        else:
            Xhat = X_mu
        return Xhat
