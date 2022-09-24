import tensorflow as tf


class AbsSumtoOne(tf.keras.constraints.Constraint):
    def __call__(self, w: tf.Variable) -> tf.Variable:
        w = tf.abs(w)

        w = w / tf.reshape(
            tf.reduce_sum(w, axis=1),
            (-1, 1),
        )
        return tf.nn.relu(w - 1e-3)
