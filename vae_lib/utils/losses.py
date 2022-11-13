import tensorflow as tf


def mean_absolute_error(
    x_in: tf.Tensor, x_out: tf.Tensor, reduce: bool = True
) -> tf.Tensor:
    mae = tf.abs(x_in - x_out)
    if reduce:
        mae = reduce_loss(mae)
    return mae


def mean_squared_error(
    x_in: tf.Tensor, x_out: tf.Tensor, reduce: bool = True
) -> tf.Tensor:
    mse = tf.square(x_in - x_out)
    if reduce:
        mse = reduce_loss(mse)
    return mse


def reduce_loss(loss: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))


def kl_divergence(mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
    return -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
