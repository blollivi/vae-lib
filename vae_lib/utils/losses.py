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


def kl_divergence(mu: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
    return -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))


def compute_kernel(x: tf.Tensor, y: tf.Tensor):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def maximum_mean_discrepancy(x: tf.Tensor, y: tf.Tensor):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)