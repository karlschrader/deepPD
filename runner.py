import logging
import tensorflow as tf
from deep_bsde_solver import DeepBSDESolver
from dual_solver import DualSolver


class Runner():
    """
    This class encapsulates the algorthmis core and tensorflow components
    and provides methods to run tasks based on a BSDE and configurations.
    """

    def __init__(self, bsde_class, bsde_config, run_name=""):
        self._bsde = bsde_class(bsde_config)
        self._deep_model = DeepBSDESolver(self._bsde, run_name)
        self._dual_model = DualSolver(self._bsde, run_name)
        self._deep_graph = None
        self._dual_graph = None
        self._sess = None

    def run_deep_nn(self):
        with self._deep_graph.as_default():
            y_0 = self._deep_model.get_y0(self._sess)
        return y_0

    def build_deep_nn(self, deep_config):
        logging.info("Building Deep model...")
        tf.reset_default_graph()
        self._deep_graph = tf.Graph()
        with self._deep_graph.as_default():
            self._deep_model.build(deep_config)
        self._sess = tf.Session(graph=self._deep_graph)
        logging.info("Building Deep model done.")

    def restore_deep_nn(self, filename):
        with self._deep_graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self._sess, filename)
        logging.info("Restore successful.")

    def restore_dual_nn(self, filename):
        with self._dual_graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self._sess, filename)
        logging.info("Restore successful.")

    def train_deep_nn(self):
        logging.info("Training Deep model...")
        with self._deep_graph.as_default():
            self._deep_model.train(self._sess)
            y_0 = self._deep_model.get_y0(self._sess)
        logging.info("Training Deep model done, %.5e", y_0)
        return y_0

    def save_deep_nn(self, save_dir="./saves/last_run"):
        logging.info("Saving Deep model...")
        if not self._deep_graph is None:
            with self._deep_graph.as_default():
                saver = tf.train.Saver()
                saver.save(self._sess, save_dir + "_deep")
        logging.info("Saving Deep model done.")

    def save_dual_nn(self, save_dir="./saves/last_run"):
        logging.info("Saving Dual model...")
        if not self._dual_graph is None:
            with self._dual_graph.as_default():
                saver = tf.train.Saver()
                saver.save(self._sess, save_dir + "_dual")
        logging.info("Saving Dual model done.")

    def run_primal(self, primal_config):
        logging.info("Running Primal...")
        primal, _ = self._deep_model.primal(self._sess, primal_config)
        logging.info("Primal Bound: %.5e", primal)
        return primal

    def build_dual_nn(self, dual_config, deep_config):
        logging.info("Building Dual...")
        weights = self._deep_model.get_matrix_weights(self._sess)
        tf.reset_default_graph()
        self._dual_graph = tf.Graph()
        with self._dual_graph.as_default():
            self._dual_model.build(dual_config, deep_config, weights)
        self._sess = tf.Session(graph=self._dual_graph)
        logging.info("Building Dual done.")

    def train_dual_nn(self):
        logging.info("Starting Dual training...")
        with self._dual_graph.as_default():
            self._dual_model.train(self._sess)
        logging.info("Dual training done.")

    def run_dual(self):
        logging.info("Running Dual...")
        with self._dual_graph.as_default():
            dual = self._dual_model.dual(self._sess)
        logging.info("Dual Bound: %.5e", dual)
        return dual
