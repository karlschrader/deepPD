import logging
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from feed_forward_model import *


class DeepBSDESolver(FeedForwardModel):
    """
    Implementation of the Deep BSDE solver and Primal bound using tensorflow.
    """

    def __init__(self, bsde, run_name):
        super().__init__(bsde, run_name + "_deep")

    def train(self, sess):
        """
        Train the Deep BSDE solver

        Args:
            sess (tf.Session): tensorflow session to use for training.

        Returns:
            _loss (float): loss after the last step of training
            _y_init (float): value of Y_0 after the last run
        """

        # validation sets
        dw_valid, x_valid = self._bsde.sample(self._deep_config.valid_size)

        # since we still batches during validation, batch norm can stay on
        feed_dict_valid = {
            self._dw: dw_valid,
            self._x: x_valid,
            self._is_training: False
        }

        sess.run(tf.global_variables_initializer())

        # used to dump training progress
        test_writer = tf.summary.FileWriter(self.tb_dir, sess.graph)

        # do sgd
        for step in tqdm(range(1, self._deep_config.num_iterations + 1)):
            # check learning progress
            if step % self._deep_config.logging_frequency == 0:
                _, _, summary = sess.run(
                    [self._loss, self._y_init, self._merged],
                    feed_dict=feed_dict_valid)
                test_writer.add_summary(summary, step)

            # run one training batch
            dw_train, x_train = self._bsde.sample(self._deep_config.batch_size)
            sess.run(
                self._train_ops,
                feed_dict={
                    self._dw: dw_train,
                    self._x: x_train,
                    self._is_training: True
                })
        return (self._loss, self._y_init)

    def build(self, deep_config):
        """
        Build the neural network for the Deep BSDE solver and Primal bound

        Args:
            deep_config (DeepConfig): Configuration instructions for the neural network
        """

        self._deep_config = deep_config
        start_time = time.time()

        #Generate the discrete time steps
        time_stamp = np.arange(
            0, self._bsde.num_time_interval) * self._bsde.delta_t

        #inputs to the neural network. Will be filled by feed_dicts later on
        self._dw = tf.placeholder(
            TF_DTYPE, [None, self._bsde.dim, self._bsde.num_time_interval],
            name='dW')
        self._x = tf.placeholder(
            TF_DTYPE, [None, self._bsde.dim, self._bsde.num_time_interval + 1],
            name='X')
        self._is_training = tf.placeholder(tf.bool)

        #inital guess for Y_0. Might be available due to other approximations
        self._y_init = tf.Variable(
            tf.random_uniform(
                [1],
                minval=self._deep_config.y_init_range[0],
                maxval=self._deep_config.y_init_range[1],
                dtype=TF_DTYPE),
            name="y_init")
        tf.summary.scalar('Y_0', self._y_init[0])

        # variable for Z_0
        z_init = tf.Variable(
            tf.random_uniform(
                [1, self._bsde.dim], minval=-.1, maxval=.1, dtype=TF_DTYPE))

        # init batch-sized vectors
        all_one_vec = tf.ones(
            shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE)
        y = all_one_vec * self._y_init
        z = tf.matmul(all_one_vec, z_init)

        # init integral approximations for Primal bound
        i = all_one_vec * 0.0
        q = all_one_vec * 0.0
        #old_z = all_one_vec * 0.0

        # will hold references to all used matrix weights
        self._matrix_weights = []

        with tf.variable_scope('forward'):

            for t in range(0, self._bsde.num_time_interval - 1):
                # Primal update
                a_star = self._bsde.max_a_tf(y)
                q = q + tf.exp(i) * self._bsde.f_star_tf(
                    a_star) * self._bsde.delta_t
                i = i + self._bsde.delta_t * a_star

                # note that all the examples use  sigma(t, X) independent of t and X,
                # so those calculations are already folded into the generation of dw

                # Deep BSDE update
                y = y - self._bsde.delta_t * (self._bsde.f_tf(
                    time_stamp[t], self._x[:, :, t], y, z)) + tf.reduce_sum(
                        z * self._dw[:, :, t], 1, keepdims=True)
                #+ tf.reduce_sum(
                #     tf.get_variable(
                #         'skip_connection_%u' % t,
                #         self._bsde.dim,
                #         TF_DTYPE,
                #         initializer=tf.random_normal_initializer(0.0, stddev=0.1, dtype=TF_DTYPE)) * old_z *
                #     self._dw[:, :, t],
                #     1,
                #     keepdims=True)
                #generate next gradient z via the nn
                #old_z = z

                #generate next subnet
                z, weight = self._subnetwork(self._x[:, :, t + 1], t + 1,
                                             self._deep_config.num_hiddens)
                z = z / self._bsde.dim
                self._matrix_weights.append(weight)

            # Last primal update
            a_star = self._bsde.max_a_tf(y)
            q = q + tf.exp(i) * self._bsde.f_star_tf(
                a_star) * self._bsde.delta_t
            i = i + self._bsde.delta_t * a_star

            # Primal bound averaged over batch
            self._primal = tf.reduce_mean(
                tf.exp(i) * self._bsde.g_tf(self._bsde.total_time, self.
                                            _x[:, :, -1]) - q)

            # Last Deep BSDE Update
            y = y - self._bsde.delta_t * self._bsde.f_tf(
                time_stamp[-1], self._x[:, :, -2], y, z) + tf.reduce_sum(
                    z * self._dw[:, :, -1], 1, keepdims=True)

            # Calculate loss
            delta = y - self._bsde.g_tf(self._bsde.total_time,
                                        self._x[:, :, -1])
            self._loss = tf.reduce_mean(
                tf.where(
                    tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                    2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP**2))
            tf.summary.scalar('primal_loss', self._loss)

        self._merged = tf.summary.merge_all()

        #sgd setup
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0),
            trainable=False,
            dtype=tf.int32)
        learning_rate = tf.train.piecewise_constant(
            global_step, self._deep_config.lr_boundaries,
            self._deep_config.lr_values)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self._loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=global_step,
            name='train_step')
        all_ops = [apply_op] + self._extra_train_ops
        self._train_ops = tf.group(*all_ops)
        self._t_build = time.time() - start_time

    def primal(self, sess, primal_config):
        """
        Calculate the Primal bound

        Args:
            sess (tf.Session): tensorflow session to use
            primal_config (PrimalConfig): Configuration instructions for the primal bound

        Returns:
           float: Primal bound
        """
        start_time = time.time()
        num_batches = np.ceil(primal_config.primal_num_total_samples /
                              primal_config.primal_batch_size).astype(
                                  np.int32)

        # Primal bound of the individual batches
        primals = np.zeros(shape=[num_batches])

        for batch in tqdm(range(num_batches)):
            dw_train, x_train = self._bsde.sample(
                primal_config.primal_batch_size)
            primals[batch] = sess.run(
                self._primal,
                feed_dict={
                    self._dw: dw_train,
                    self._x: x_train,
                    self._is_training: False
                })
            if (batch % self._deep_config.logging_frequency ==
                    self._deep_config.logging_frequency - 1):
                logging.debug("Batch: %5u,    primal: %.4e",
                              (batch + 1, np.mean(primals[:batch])))

        elapsed_time = time.time() - start_time
        return np.mean(primals), (num_batches * primal_config.primal_batch_size
                                  / elapsed_time).astype(np.int64)

    def get_y0(self, sess):
        """
        Return the current Y_0 approximation

        Args:
            sess (tf.Session): tensorflow session to use

        Returns:
           float: current Y_0 approximation
        """
        return sess.run(self._y_init)[0]

    def get_matrix_weights(self, sess):
        """
    	Get the matrix weights used in neural network    	

        Args:
            sess (tf.Session): tensorflow session to use

        Returns:
           np.array(size=[num_timesteps, layer_count]): matrix weights
        """
        return sess.run(self._matrix_weights)
