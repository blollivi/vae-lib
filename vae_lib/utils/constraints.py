import tensorflow as tf


class AbsSumtoOne(tf.keras.constraints.Constraint):
    def __call__(self, w: tf.Variable) -> tf.Variable:
        w = tf.abs(w)

        w = w / tf.reshape(
            tf.reduce_sum(w, axis=1),
            (-1, 1),
        )
        return w


class BetweenZeroAndOne(tf.keras.constraints.Constraint):
    def __call__(self, w: tf.Variable) -> tf.Variable:
        min = tf.reduce_min(w, axis=0)
        max = tf.reduce_max(w, axis=0)
        w = (w - min) / (max - min)

        return w


class ToBinary(AbsSumtoOne):
    def __call__(self, w: tf.Variable, thd: float = 0.5) -> tf.Variable:
        w = super().__call__(w)
        return tf.cast(w > thd, tf.float32)