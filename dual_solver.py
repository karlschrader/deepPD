import time
#import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from feed_forward_model import *


class DualSolver(FeedForwardModel):
    """
    Implementation of the Dual bound using tensorflow.
    """

    def __init__(self, bsde, run_name):
        super().__init__(bsde, run_name + "_dual")

    def train(self, sess):
        """
        Train the Dual bound network

        Args:
            sess (tf.Session): tensorflow session to use for training.

        Returns:
            _dual_loss (float): loss after the last learning step
        """
        # validation sets
        dw_valid, x_valid = self._bsde.sample(self._dual_config.valid_size)

        # since we still batches during validation, batch norm can stay on
        feed_dict_valid = {
            self._dw: dw_valid,
            self._x: x_valid,
            self._is_training: False
        }

        test_writer = tf.summary.FileWriter(self.tb_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        # do sgd
        for step in tqdm(range(1, self._dual_config.num_iterations + 1)):
            # check learning progress
            if step % self._dual_config.logging_frequency == 0:
                _, summary = sess.run(
                    [self._dual_loss, self._dual_merged],
                    feed_dict=feed_dict_valid)
                test_writer.add_summary(summary, step)

            # run one training batch
            dw_train, x_train = self._bsde.sample(self._dual_config.batch_size)
            sess.run(
                self._dual_train_ops,
                feed_dict={
                    self._dw: dw_train,
                    self._x: x_train,
                    self._is_training: True
                })
        return self._dual_loss

    def build(self, dual_config, deep_config, weights):
        """
        Build the neural network for the Dual bound

        Args:
            dual_config (DualConfig): Configuration instructions for the neural network
            deep_config (DeepConfig): Configuration instructions from the primal network. 
            	Needed to reuse some of the learned weights
            weights (np.array(size=[num_timesteps, layer_count])): Matrix weights used in 
            	the primal network
        """
        self._dual_config = dual_config
        self._extra_train_ops = []
        start_time = time.time()

        with tf.variable_scope('dual'):
            #inputs to the neural network. Will be filled by feed_dicts later on
            self._dw = tf.placeholder(
                TF_DTYPE, [None, self._bsde.dim, self._bsde.num_time_interval],
                name='dW')
            self._x = tf.placeholder(
                TF_DTYPE,
                [None, self._bsde.dim, self._bsde.num_time_interval + 1],
                name='X')
            self._is_training = tf.placeholder(tf.bool)

            # Last (or first) value of u = U_T
            u = self._bsde.g_tf(self._bsde.total_time, self._x[:, :, -1])

            with tf.variable_scope('forward'):
                for t in range(self._bsde.num_time_interval - 1, 0, -1):
                    u = u + self._bsde.f_mini_tf(u) * self._bsde.delta_t

                    # reuse weights or initialize fresh
                    if tf.app.flags.FLAGS.clone_deep_bsde_weights:
                        z = self._clone_subnetwork(self._x[:, :, t], t,
                                                   deep_config.n_layer,
                                                   weights) / self._bsde.dim
                    else:
                        z, _ = self._subnetwork(self._x[:, :, t], t,
                                                deep_config.num_hiddens)
                        z = z / self._bsde.dim

                    u = u - tf.reduce_sum(
                        z * self._dw[:, :, t], 1, keepdims=True)

                # final step needs no z term
                u = u + self._bsde.f_mini_tf(u) * self._bsde.delta_t
                self._dual_loss = tf.reduce_mean(u)
                tf.summary.scalar('dual_loss', self._dual_loss)

            # sgd setup
            self._dual_merged = tf.summary.merge_all()
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0),
                trainable=False,
                dtype=tf.int32)
            learning_rate = tf.train.piecewise_constant(
                global_step, self._dual_config.lr_boundaries,
                self._dual_config.lr_values)
            trainable_variables = tf.trainable_variables()
            grads = tf.gradients(self._dual_loss, trainable_variables)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            apply_op = optimizer.apply_gradients(
                zip(grads, trainable_variables),
                global_step=global_step,
                name='dual_train_step')
            all_ops = [apply_op] + self._extra_train_ops
            self._dual_train_ops = tf.group(*all_ops)
            self._t_build_dual = time.time() - start_time

    def dual(self, sess):
        """
        Calculate the Dual bound

        Args:
            sess (tf.Session): tensorflow session to use

        Returns:
           float: Primal bound
        """
        sess.run(tf.global_variables_initializer())

        num_batches = np.ceil(self._dual_config.num_total_samples /
                              self._dual_config.dual_batch_size).astype(
                                  np.int32)
        duals = np.zeros(shape=[num_batches])

        for batch in tqdm(range(num_batches)):
            dw_train, x_train = self._bsde.sample(
                self._dual_config.dual_batch_size)
            duals[batch] = sess.run(
                self._dual_loss,
                feed_dict={
                    self._dw: dw_train,
                    self._x: x_train,
                    self._is_training: False
                })

        return np.mean(duals)
