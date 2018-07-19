from collections import namedtuple
import numpy as np

# Configuration clases used for the various stages of the algorithm
ProblemConfig = namedtuple('ProblemConfig', 'total_time num_time_interval dim')
DeepConfig = namedtuple(
    'DeepConfig',
    'n_layer num_hiddens batch_size valid_size num_iterations logging_frequency y_init_range lr_values lr_boundaries'
)
PrimalConfig = namedtuple('PrimalConfig',
                          'primal_batch_size primal_num_total_samples')
DualConfig = namedtuple(
    'DualConfig',
    'batch_size valid_size logging_frequency lr_values lr_boundaries num_iterations dual_batch_size num_total_samples'
)

# For most problems, it is sufficient to clone a base config and only modify selected values.
problemConfigBase = ProblemConfig(dim=50, total_time=0.3, num_time_interval=10)
deepConfigBase = DeepConfig(
    n_layer=4,
    num_hiddens=[0, 10, 10, 0],
    batch_size=64,
    valid_size=256,
    num_iterations=3000,
    logging_frequency=50,
    y_init_range=[0.1, 0.4],
    lr_values=list(np.array([5e-4, 5e-5])),
    lr_boundaries=[2000])
primalConfigBase = PrimalConfig(
    primal_batch_size=512, primal_num_total_samples=10**6)
dualConfigBase = DualConfig(
    batch_size=64,
    valid_size=256,
    logging_frequency=50,
    lr_values=list(np.array([5e-4, 5e-5])),
    #lr_values=list(np.array([5e-4, 5e-4, 5e-4])),
    lr_boundaries=[2000],
    num_iterations=3000,
    dual_batch_size=512,
    num_total_samples=10**6)

#Minimal set of configurations used for testing if the networks build correctly
problemConfigTest = ProblemConfig(dim=10, total_time=1, num_time_interval=2)
deepConfigTest = deepConfigBase._replace(num_iterations=200)
primalConfigTest = primalConfigBase._replace(primal_num_total_samples=1024)
dualConfigTest = dualConfigBase._replace(
    num_iterations=200, num_total_samples=1024)
