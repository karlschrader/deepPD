import logging
import time
import os
import numpy as np
import tensorflow as tf
from runner import Runner
import configs
import bsdes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('tensorboard_dir', './train/',
                           """Directory for tensorboard log data.""")
tf.app.flags.DEFINE_boolean(
    'clone_deep_bsde_weights', False,
    """Initialize dual nn with the weights of the primal stage.""")


def dim_test():
    '''
    Stage 1 of analysing impact of dimensions of runtime.

    Only tests training speeds, and saves models to disk.
    '''
    dims = np.array([1, 25, 50, 75, 100, 250, 500, 750, 1000])
    speeds = {}
    for dimCount in dims:
        logging.info("Starting run for %4u dimensions." % dimCount)
        problem_config = configs.problemConfigBase._replace(dim=dimCount)
        runner = Runner(bsdes.Squared, problem_config, "Dim_%04u" % dimCount)

        start_time = time.time()
        runner.build_deep_nn(configs.deepConfigBase)
        deep_y0 = runner.train_deep_nn()
        elapsed_deep = time.time() - start_time
        runner.save_deep_nn("./saves/dimTest_%04u_dims" % dimCount)

        # start_time = time.time()
        # primal_y0 = runner.run_primal(configs.primalConfigBase)
        # elapsed_primal = time.time() - start_time

        start_time = time.time()
        runner.build_dual_nn(configs.dualConfigBase, configs.deepConfigBase)
        runner.train_dual_nn()
        elapsed_dual_train = time.time() - start_time
        runner.save_dual_nn("./saves/dimTest_%04u_dims" % dimCount)

        # start_time = time.time()
        # dual_y0 = runner.run_dual()
        # elapsed_dual_run = time.time() - start_time

        #logging.info("Primal: %.5e Deep: %.5e Dual: %.5e" % (primal_y0, deep_y0, dual_y0))
        speeds[dimCount] = (elapsed_deep, elapsed_dual_train)
    np.save("speedtest_half.dump", speeds)


def dim_test2():
    '''
    Stage 2 of analysing impact of dimensions of runtime.

    Loads previously generated models and tests Primal and Dual sampling.    
    '''

    dims = np.array([1, 25, 50, 75, 100, 250, 500, 750, 1000])
    speeds = {}
    for dimCount in dims:
        logging.info("Starting run for %4u dimensions." % dimCount)
        problem_config = configs.problemConfigBase._replace(dim=dimCount)
        runner = Runner(bsdes.Squared, problem_config, "Dim_%04u" % dimCount)

        runner.build_deep_nn(configs.deepConfigBase)
        runner.restore_deep_nn("./saves/dimTest_%04u_dims_deep" % dimCount)

        start_time = time.time()
        primal_y0 = runner.run_primal(configs.primalConfigBase)
        elapsed_primal = time.time() - start_time

        runner.build_dual_nn(configs.dualConfigBase, configs.deepConfigBase)
        runner.restore_dual_nn("./saves/dimTest_%04u_dims_dual" % dimCount)

        start_time = time.time()
        dual_y0 = runner.run_dual()
        elapsed_dual_run = time.time() - start_time

        #logging.info("Primal: %.5e Deep: %.5e Dual: %.5e" % (primal_y0, deep_y0, dual_y0))
        speeds[dimCount] = (elapsed_primal, elapsed_dual_run)
    np.save("speedtest_other_half.dump", speeds)


def tiv_test():
    '''
    Test impact of number of time intervals on convergence.

    Analysis is done using tensorboard.
    Also records speeds to verify linear dependence.
    '''
    stepping = np.array([5, 10, 15, 20, 25, 30])
    speeds = {}
    for steps in stepping:
        tf.set_random_seed(4242424242)
        np.random.seed(4242424242)
        logging.info("Starting run for %02u steps." % steps)
        problem_config = configs.problemConfigBase._replace(
            num_time_interval=steps)
        runner = Runner(bsdes.Squared, problem_config, "Step_%04u" % steps)

        start_time = time.time()
        runner.build_deep_nn(configs.deepConfigBase)
        deep_y0 = runner.train_deep_nn()
        elapsed_deep = time.time() - start_time
        runner.save_deep_nn("./saves/tivTest_%04u_steps" % steps)

        start_time = time.time()
        primal_y0 = runner.run_primal(configs.primalConfigBase)
        elapsed_primal = time.time() - start_time

        start_time = time.time()
        runner.build_dual_nn(configs.dualConfigBase, configs.deepConfigBase)
        runner.train_dual_nn()
        elapsed_dual_train = time.time() - start_time
        runner.save_dual_nn("./saves/tivTest_%04u_steps" % steps)

        start_time = time.time()
        dual_y0 = runner.run_dual()
        elapsed_dual = time.time() - start_time

        #logging.info("Primal: %.5e Deep: %.5e Dual: %.5e" % (primal_y0, deep_y0, dual_y0))
        speeds[steps] = (elapsed_deep, elapsed_primal, elapsed_dual_train,
                         elapsed_dual)
    np.save("tivtest.dump", speeds)


def y0_impact_test():
    '''
    Test impact of inital value of y_0 on convergence of the deep BSDE solver.

    Analysis is done using tensorboard.
    '''
    y0_guesses = np.array([0, 0.1, 0.2, 0.26, 0.3, 0.4])
    for guess in y0_guesses:
        tf.set_random_seed(4242424242)
        np.random.seed(4242424242)
        logging.info("Starting run for y_0 = %.2e" % guess)
        runner = Runner(bsdes.Squared, configs.problemConfigBase,
                        "guess_%u" % (guess * 100))

        start_time = time.time()
        deep_config = configs.deepConfigBase._replace(
            y_init_range=[guess, guess])
        runner.build_deep_nn(deep_config)
        deep_y0 = runner.train_deep_nn()


def depth_test():
    '''
    Test impact of different subnetwork structures on convergence.

    Analysis is done using tensorboard.
    Also records speeds to verify linear dependence.
    '''
    depths = np.array([3, 3, 4, 4, 5, 5])
    structures = np.array([[0, 0, 0], [0, 10, 0], [0, 0, 0, 0], [0, 10, 10, 0],
                           [0, 0, 0, 0, 0], [0, 10, 10, 10, 0]])

    speeds = {}
    for (depth, structure) in zip(depths, structures):
        tf.set_random_seed(4242424242)
        np.random.seed(4242424242)
        logging.info("Starting run for" + str(structure))
        runner = Runner(bsdes.Squared, configs.problemConfigBase,
                        "Depth_%s" % str(structure))

        deep_config = configs.deepConfigBase._replace(
            n_layer=depth, num_hiddens=structure)

        start_time = time.time()
        runner.build_deep_nn(deep_config)
        deep_y0 = runner.train_deep_nn()
        elapsed_deep = time.time() - start_time
        runner.save_deep_nn("./saves/depthTest_%s_steps" % str(structure))

        start_time = time.time()
        primal_y0 = runner.run_primal(configs.primalConfigBase)
        elapsed_primal = time.time() - start_time

        start_time = time.time()
        runner.build_dual_nn(configs.dualConfigBase, deep_config)
        runner.train_dual_nn()
        elapsed_dual_train = time.time() - start_time
        runner.save_dual_nn("./saves/depthTest_%s_steps" % str(structure))

        start_time = time.time()
        dual_y0 = runner.run_dual()
        elapsed_dual = time.time() - start_time

        speeds[str(structure)] = (elapsed_deep, elapsed_primal,
                                  elapsed_dual_train, elapsed_dual)
    np.save("depthtest.dump", speeds)


def linear():
    '''
    Do a full run for f(u) = u
    '''
    runner = Runner(bsdes.Linear, configs.problemConfigBase)
    runner.build_deep_nn(configs.deepConfigBase)
    deep_y0 = runner.train_deep_nn()
    runner.save_deep_nn("./saves/T1")
    #runner.restore_nn("./saves/SHACV2.ckpt")
    primal_y0 = runner.run_primal(configs.primalConfigBase)
    runner.build_dual_nn(configs.dualConfigBase, configs.deepConfigBase)
    #runner.restore_nn("./saves/SHACV2.ckpt")
    runner.train_dual_nn()
    runner.save_dual_nn("./saves/T1")
    dual_y0 = runner.run_dual()
    logging.info("Primal: %.5e Deep: %.5e Dual: %.5e" % (primal_y0, deep_y0,
                                                         dual_y0))


def square_learn():
    '''
    Do a full run for f(u) = u^2
    '''
    runner = Runner(bsdes.Squared, configs.problemConfigBase)
    runner.build_deep_nn(configs.deepConfigBase)
    deep_y0 = runner.train_deep_nn()
    #runner.restore_nn("./saves/SHACV2.ckpt")
    primal_y0 = runner.run_primal(configs.primalConfigBase)
    runner.save_deep_nn("./saves/T2")
    runner.build_dual_nn(configs.dualConfigBase, configs.deepConfigBase)
    #runner.restore_nn("./saves/SHACV2.ckpt")
    runner.train_dual_nn()
    dual_y0 = runner.run_dual()
    runner.save_dual_nn("./saves/T2")
    logging.info("Primal: %.5e Deep: %.5e Dual: %.5e" % (primal_y0, deep_y0,
                                                         dual_y0))


def square_restore():
    '''
    Do only Dual sampling, using networks restored from disk.
    '''
    runner = Runner(bsdes.Squared, configs.problemConfigBase)
    runner.build_deep_nn(configs.deepConfigBase)
    #deep_y0 = runner.train_deep_nn()
    runner.restore_deep_nn("./saves/T2_deep")
    #deep_y0 = runner.run_deep_nn()

    #primal_y0 = runner.run_primal(configs.primalConfigBase)

    runner.build_dual_nn(configs.dualConfigBase, configs.deepConfigBase)
    #runner.train_dual_nn()
    runner.restore_dual_nn("./saves/T2_dual")
    dual_y0 = runner.run_dual()
    #logging.info("Primal: %.5e Deep: %.5e Dual: %.5e" % (primal_y0, deep_y0, dual_y0))


def main():
    logging.basicConfig(
        level=logging.INFO, format='%(levelname)-6s %(message)s')
    if (not os.path.exists('./saves')):
        os.mkdir('./saves')
    if (not os.path.exists(FLAGS.tensorboard_dir)):
        os.mkdir(FLAGS.tensorboard_dir)
    square_learn()


if __name__ == '__main__':
    main()
