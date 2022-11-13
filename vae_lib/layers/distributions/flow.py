import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfpd = tfp.distributions



class TransformedGaussianDistribution(tfpd.TransformedDistribution):
    def __init__(self, output_dim: int, n_layers: int) -> None:
        self.flow_layers  = [
            PlanarFlow(output_dim) for _ in range(n_layers)
        ]
        self.flow = tfb.Chain(
            bijectors=list(reversed(self.flow_layers)), name='chain_of_planar'
        )

        super().__init__(
            distribution=tfpd.MultivariateNormalDiag(
                loc=tf.zeros(output_dim),
                scale_diag=tf.ones(output_dim)
            ),
            bijector=self.flow
        )


class PlanarFlow(tfb.Bijector, tf.Module):
    '''
    Implementation of Planar Flow .
    '''

    def __init__(self, output_dim, name="planar_flow"):
        super().__init__(
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            name=name)

        self.event_ndims = 1

        self.u = tf.Variable(
            np.random.uniform(-1., 1., size=(output_dim)),
            name='u',
            dtype=tf.float32,
            trainable=True
        )
        self.w = tf.Variable(
            np.random.uniform(-1., 1., size=(output_dim)),
            name='w',
            dtype=tf.float32,
            trainable=True
        )
        self.b = tf.Variable(
            np.random.uniform(-1., 1., size=(1)),
            name='b',
            dtype=tf.float32,
             trainable=True
        )

    def h(self, y):
        return tf.math.tanh(y)

    def h_prime(self, y):
        return 1.0 - tf.math.tanh(y) ** 2.0

    def alpha(self):
        wu = tf.tensordot(self.w, self.u, 1)
        m = -1.0 + tf.nn.softplus(wu)
        return m - wu

    def _u(self):
        if tf.tensordot(self.w, self.u, 1) <= -1:
            alpha = self.alpha()
            z_para = tf.transpose(
                alpha * self.w / tf.math.sqrt(tf.reduce_sum(self.w ** 2.0)))
            self.u.assign_add(z_para)  # self.u = self.u + z_para

    def _forward_func(self, zk):
        inter_1 = self.h(tf.tensordot(zk, self.w, 1) + self.b)
        return tf.add(zk, tf.tensordot(inter_1, self.u, 0))

    def _forward(self, zk):
        return self._forward_func(zk)

    def _inverse(self, zk):
        return self._forward_func(zk)

    def _log_det_jacobian(self, zk):
        psi = tf.tensordot(self.h_prime(
            tf.tensordot(zk, self.w, 1) + self.b), self.w, 0)
        det = tf.math.abs(1.0 + tf.tensordot(psi, self.u, 1))
        return tf.math.log(det)

    def _forward_log_det_jacobian(self, zk):
        if self.case == "sampling":
            return -self._log_det_jacobian(zk)
        else:
            raise NotImplementedError(
                '_forward_log_det_jacobian is not implemented for density_estimation')

    def _inverse_log_det_jacobian(self, zk):
        return self._log_det_jacobian(zk)
