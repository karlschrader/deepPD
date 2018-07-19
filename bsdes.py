import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal


class BSDE():
    """ Abstract base class for defining a PDE by its functions used in the algorithms.

    All functions are implemented using tensorflow operations.
    """

    def __init__(self, bsde_config):
        self._bsde_config = bsde_config
        self._delta_t = (self._bsde_config.total_time + 0.0
                         ) / self._bsde_config.num_time_interval
        self._sqrt_delta_t = np.sqrt(self._delta_t)
        self._y_init = None

    def sample(self, num_sample):
        """ Generate a number of sample paths of X and W

        Args:
            num_sample (int): Number of samples to generate

        Returns:
            dw_sample (np.array(size=[num_sample, self._bsde_config.dim, self._bsde_config.num_time_interval]))
            x_sample (np.array([num_sample, self._bsde_config.dim, self._bsde_config.num_time_interval + 1]))

        """
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """ Calculate the generator function f of the PDE

        Args:
            t (float): Current time
            x (np.array(size=[self._bsde_config.dim])): Value of X_t
            y (float): Value of (the approximation of) Y_t
            z (np.array(size=[self._bsde_config.dim])): Value of (the approximation of) Z_t

        Returns:
            float

        Note that this function also works for mulitdimensional t, x, y, z, i.p. for multiple samples at once.
        """
        raise NotImplementedError

    def g_tf(self, t, x):
        """ Calculate the terminal condition g of the PDE

        Args:
            t (float): Terminal time T
            x (np.array(size=[self._bsde_config.dim])): Value of X_T

        Returns:
            float

        Note that this function also works for mulitdimensional t, x, y, z, i.p. for multiple samples at once.
        """
        raise NotImplementedError

    def f_star_tf(self, a):
        """ Calculate the value of the Legendre transform of f

        Args:
            a (float)

        Returns:
            float: f^*(a)

        Note that this function also works for mulitdimensional t, x, y, z, i.p. for multiple samples at once.
        """
        raise NotImplementedError

    def f_mini_tf(self, y):
        """ Calculate the generator function f of the PDE. Restricted to Y_t as an input.

        Args:
            y (float): Value of (the approximation of) Y_t

        Returns:
            float

        Note that this function also works for mulitdimensional y, i.p. for multiple samples at once.
        """
        raise NotImplementedError

    def max_a_tf(self, y):
        """ Calculate argmax_a{aY-f^*(a)} of the PDE.

        Args:
            y (float): Value of (the approximation of) Y_t

        Returns:
            float

        Note that this function also works for mulitdimensional y, i.p. for multiple samples at once.
        """
        raise NotImplementedError

    @property
    def y_init(self):
        return self._y_init

    @property
    def dim(self):
        return self._bsde_config.dim

    @property
    def num_time_interval(self):
        return self._bsde_config.num_time_interval

    @property
    def total_time(self):
        return self._bsde_config.total_time

    @property
    def delta_t(self):
        return self._delta_t


class Squared(BSDE):
    """
    BSDE corresponding to f(u) = u^2, dX_t = sqrt(2) dW_t, g = 0.5/(1+0.2* ||x||_2)
    """

    def __init__(self, bsde_config):
        super().__init__(bsde_config)
        self._x_init = np.zeros(self._bsde_config.dim)
        self._sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[
            num_sample, self._bsde_config.dim,
            self._bsde_config.num_time_interval
        ]) * self._sqrt_delta_t
        if self._bsde_config.dim == 1:
            dw_sample = dw_sample.reshape([
                num_sample, self._bsde_config.dim,
                self._bsde_config.num_time_interval
            ])
        x_sample = np.zeros([
            num_sample, self._bsde_config.dim,
            self._bsde_config.num_time_interval + 1
        ])
        x_sample[:, :, 0] = np.ones([num_sample, self._bsde_config.dim
                                     ]) * self._x_init
        for i in range(self._bsde_config.num_time_interval):
            x_sample[:, :, i +
                     1] = x_sample[:, :, i] + self._sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return y**2

    def g_tf(self, t, x):
        return 0.5 / (1 + 0.2 * tf.reduce_sum(tf.square(x), 1, keepdims=True))

    def max_a_tf(self, y):
        return 2 * y

    def f_star_tf(self, a):
        return (a**2) / 4

    def f_mini_tf(self, y):
        return y**2


class Linear(BSDE):
    """
    BSDE corresponding to f(u) = u, dX_t = sqrt(2) dW_t, g = 0.5/(1+0.2* ||x||_2)
    """

    def __init__(self, bsde_config):
        super().__init__(bsde_config)
        self._x_init = np.zeros(self._bsde_config.dim)
        self._sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[
            num_sample, self._bsde_config.dim,
            self._bsde_config.num_time_interval
        ]) * self._sqrt_delta_t
        if self._bsde_config.dim == 1:
            dw_sample = dw_sample.reshape([
                num_sample, self._bsde_config.dim,
                self._bsde_config.num_time_interval
            ])
        x_sample = np.zeros([
            num_sample, self._bsde_config.dim,
            self._bsde_config.num_time_interval + 1
        ])
        x_sample[:, :, 0] = np.ones([num_sample, self._bsde_config.dim
                                     ]) * self._x_init
        for i in range(self._bsde_config.num_time_interval):
            x_sample[:, :, i +
                     1] = x_sample[:, :, i] + self._sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return y

    def g_tf(self, t, x):
        return 0.5 / (1 + 0.2 * tf.reduce_sum(tf.square(x), 1, keepdims=True))

    def max_a_tf(self, y):
        return 1.

    def f_star_tf(self, a):
        return 0.

    def f_mini_tf(self, y):
        return y
