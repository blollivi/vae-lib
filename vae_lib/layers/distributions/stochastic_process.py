import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions

from vae_lib.layers.base import BaseLayer

from .types import DeepGaussianDistributionParams
from .deep_gaussian_distribution import DeepGaussianDistribution, GaussianSampler
from .flow import TransformedGaussianDistribution


class RecurrentGaussianDistribution(BaseLayer):
    def __init__(
        self,
        output_dim: int,
        gaussian_regressor_params: DeepGaussianDistributionParams,
        n_flow_layers: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.gaussian_sampler = GaussianSampler()
        self.gaussian_regressor = DeepGaussianDistribution(**gaussian_regressor_params)
        self.use_flow = n_flow_layers > 0

        if self.use_flow:
            self.prior = TransformedGaussianDistribution(self.output_dim, n_flow_layers)
        else:
            self.prior = tfpd.MultivariateNormalDiag(
                loc=tf.zeros(self.output_dim),
                scale_diag=tf.ones(self.output_dim)
            )
            

    def step(
        self,
        previous_Z: tf.Tensor,
        current_state: tf.Tensor
    ) -> tf.Tensor:
        """_summary_

        Parameters
        ----------
        previous_Z : tf.Tensor
            Tensor with shape [batch_size, output_dim]
        current_state : tf.Tensor
           Tensor with shape [batch_size, hidden_units]

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Pair of tensors, with shape [batch_size, output_dim]
        """

        gaussian_regressor_inputs = tf.concat(
            [previous_Z, current_state],
            axis=-1
        )

        current_Z_mean, current_Z_logvar = self.gaussian_regressor(
            gaussian_regressor_inputs
        )
        current_Z = self.gaussian_sampler(current_Z_mean, current_Z_logvar)

        if self.use_flow:
            # Apply flow transformation
            current_Z = self.prior.flow(current_Z)

        return current_Z
    

    def call(self, hidden_states: tf.Tensor):
        """_summary_

        Parameters
        ----------
        hidden_states : tf.Tensor
           Tensor, with shape [batch_size, timesteps, hidden_units].

        Returns
        -------
        tf.Tensor
            Pair of tensors, with shape [batch_size, timesteps, output_dim]
        """

        batch_size = tf.shape(hidden_states[0])[0]
        previous_Z_init = tf.zeros(shape=(batch_size, self.output_dim))

        # Iteratively apply computes X_t from hidden_state_t and X_{t-1}
        output_sequence = tf.scan(
            self.step,
            elems=hidden_states,
            initializer=previous_Z_init,
            back_prop=False
        )

        return output_sequence


class StochasticProcess(BaseLayer):

    def __init__(
        self,
        output_dim: int,
        num_hidden_units: int,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_states_estimator = tf.keras.layers.GRU(
            units=num_hidden_units
        )

        self.recurrent_gaussian_distribution = RecurrentGaussianDistribution(output_dim)

    def call(self, input_sequence: tf.Tensor) -> tf.Tensor:
        """_summary_

        Parameters
        ----------
        X_sequence : tf.Tensor
            Tensor, with shape [batch_size, timesteps, x_dim].

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Pair of tensors, with shape [batch_size, timesteps, z_dim]
        """

        hidden_states = self.hidden_states_estimator(input_sequence)

        output_sequence = self.recurrent_gaussian_distribution(hidden_states)

        return output_sequence
