import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

TF_DTYPE = tf.float64
MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


class FeedForwardModel():
    """
    Abstract class for creating neural networks.

    Offers functions to build or clone individual layers of complete networks
    """

    def __init__(self, bsde, run_name):
        self._bsde = bsde

        # ops for statistics update of batch normalization
        self._extra_train_ops = []

        self.tb_dir = tf.app.flags.FLAGS.tensorboard_dir + run_name + "_" + datetime.now(
        ).strftime('%Y_%m_%d_%H_%M_%S')
        os.mkdir(self.tb_dir)

    def _clone_subnetwork(self, input_, timestep, layer_count, weights):
        """
        Clone a neural network, using the same weights as the source networks.

        Args:
            input_ (Tensor): Input of the neural network that will be build
            timestep (float): Time index, used for tensor names
            layer_count (int): number of layers in the neural network that should be cloned
            weights (np.array(size=[num_timesteps, layer_count]))

        Returns:
            Tensor: Output of the last layer of the neural network
        """
        with tf.variable_scope(str(timestep)):
            hiddens = self._batch_norm(input_, name='path_input_norm')
            for i in range(1, layer_count - 1):
                hiddens = self._copy_batch_layer(hiddens, 'layer_{}'.format(i),
                                                 i, timestep, weights)
            output = self._copy_batch_layer(hiddens, 'final_layer',
                                            layer_count - 1, timestep, weights)
        return output

    def _subnetwork(self, input_, timestep, num_hiddens):
        """
        Generate a neural network

        Args:
            input_ (Tensor): Input of the neural network that will be build
            timestep (float): Time index, used for tensor name
            num_hiddens (np.array(size=[layer_count])): Specifies the number
                of additional dimensions for each layer of the neural network.

        Returns:
            Tensor: Output of the last layer of the neural network
        """
        matrix_weights = []
        with tf.variable_scope(str(timestep)):
            # input norm
            hiddens = self._batch_norm(input_, name='path_input_norm')
            for i in range(1, len(num_hiddens) - 1):
                hiddens, weight = self._dense_batch_layer(
                    hiddens,
                    num_hiddens[i] + self._bsde.dim,
                    activation_fn=tf.nn.relu,
                    layer_name='layer_{}'.format(i),
                )
                matrix_weights.append(weight)
            # last layer without relu
            output, weight = self._dense_batch_layer(
                hiddens,
                num_hiddens[-1] + self._bsde.dim,
                activation_fn=None,
                layer_name='final_layer',
            )
            matrix_weights.append(weight)
        return output, matrix_weights

    def _dense_batch_layer(self,
                           input_,
                           output_size,
                           activation_fn=None,
                           stddev=5.0,
                           layer_name="linear"):
        """
        Generate one fully connected layer

        Args:
            input_ (Tensor): Input of layer
            output_size (int): Number of outputs this layer should have

        KwArgs:
            activation_fn (Function): activation function for the neurons in
                this layer. Will usually be ReLU, but can be left blank for the last layer.
            stddev (float): stddev to use for the initial distribution of weights in this layer
            layer_name (string): tensorflow name used for the variables in this layer

        Returns:
            Tensor: Output of the layer
            tf.Variable: Reference to the used Matrix weight
        """
        with tf.variable_scope(layer_name):
            shape = input_.get_shape().as_list()
            weight = tf.get_variable(
                'Matrix', [shape[1], output_size],
                TF_DTYPE,
                tf.random_normal_initializer(
                    stddev=stddev / np.sqrt(shape[1] + output_size)))
            # matrix weight
            hiddens = tf.matmul(input_, weight)
            #batch norm
            hiddens_bn = self._batch_norm(hiddens)
        if activation_fn:
            return activation_fn(hiddens_bn), weight
        return hiddens_bn, weight

    def _copy_batch_layer(self, input_, layer_name, layer, timestep, weights):
        """
        Copy one fully connected layer, reusing the weights of the previous layer

        Args:
            input_ (Tensor): Input of layer
            layer_name (string): tensorflow name used for the variables in this layer
            layer (int): index of the layer in the current timestep
            timestep (int): index of the current timestep
            weights (np.array(size=[num_timesteps, layer_count])): weight database to copy from

        Returns:
            Tensor: Output of the layer
        """
        with tf.variable_scope(layer_name):
            # init matrix weight with matrix weights from primal stage
            weight = tf.Variable(weights[timestep - 1][layer - 1], 'Matrix')

            hiddens = tf.matmul(input_, weight)
            hiddens_bn = self._batch_norm(hiddens)
        return hiddens_bn

    def _batch_norm(self, input_, name='batch_norm'):
        """
        Batch normalize the data

        Args:
            input_ (Tensor): Input of layer

        KwArgs:
            name (string): Used as tensorflow name

        Returns:
            Tensor: Output of the layer

        See https://arxiv.org/pdf/1502.03167v3.pdf p.3
        """
        with tf.variable_scope(name):
            params_shape = [input_.get_shape()[-1]]
            beta = tf.get_variable(
                'beta',
                params_shape,
                TF_DTYPE,
                initializer=tf.random_normal_initializer(
                    0.0, stddev=0.1, dtype=TF_DTYPE))
            gamma = tf.get_variable(
                'gamma',
                params_shape,
                TF_DTYPE,
                initializer=tf.random_uniform_initializer(
                    0.1, 0.5, dtype=TF_DTYPE))
            moving_mean = tf.get_variable(
                'moving_mean',
                params_shape,
                TF_DTYPE,
                initializer=tf.constant_initializer(0.0, TF_DTYPE),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance',
                params_shape,
                TF_DTYPE,
                initializer=tf.constant_initializer(1.0, TF_DTYPE),
                trainable=False)
            # These ops will only be performed when training
            mean, variance = tf.nn.moments(input_, [0], name='moments')
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(moving_mean, mean,
                                                      MOMENTUM))
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(moving_variance,
                                                      variance, MOMENTUM))
            mean, variance = tf.cond(self._is_training,
                                     lambda: (mean, variance),
                                     lambda: (moving_mean, moving_variance))
            hiddens_bn = tf.nn.batch_normalization(input_, mean, variance,
                                                   beta, gamma, EPSILON)
            hiddens_bn.set_shape(input_.get_shape())
            return hiddens_bn
