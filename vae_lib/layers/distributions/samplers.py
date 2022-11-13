import tensorflow as tf

from vae_lib.layers.base import BaseLayer


class GaussianSampler(BaseLayer):
    """Layers that samples that takes parameters mean and logvar of a Gaussian
    and returns a sample.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        eps = tf.random.normal(shape=tf.shape(mean))
        sigma = tf.exp(logvar * 0.5)
        return mean + eps * sigma


class SpikeSlabSampler(BaseLayer):
    def __init__(self, c: float) -> None:
        super().__init__()
        self.config["c"] = c
        self.gaussian_sampler = GaussianSampler()

    def call(self, mean: tf.Tensor, logvar: tf.Tensor, logspike: tf.Tensor) -> tf.Tensor:
        gaussian_sample = self.gaussian_sampler(mean, logvar)
        eta = tf.random.uniform(shape=tf.shape(mean))
        selection = tf.nn.sigmoid(self.config["c"] * (eta + tf.exp(logspike) - 1))
        return tf.multiply(selection, gaussian_sample)