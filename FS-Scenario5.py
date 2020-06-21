"""
ModelMap

A toolkit for Model Mapping experimentation.

Authors:
Francesco Iorio <Frio@cs.toronto.edu>
Ali Hashemi
Michael Tao

License: BSD 3 clause
"""

######################################################################################################
# Filesystem experiments
#
# Uses a dataset extracted from the work published in:
# Cao, Zhen, et al.
# "Towards better understanding of black-box auto-tuning: A comparative analysis for storage systems."
# 2018 {USENIX} Annual Technical Conference ({USENIX}{ATC} 18). 2018.
#
# Categorical to numerical data conversion:
#
# inode_size_mapping
# default=128
#
# bg_count_mapping
# default=12
#
# io_scheduler
# cfq=0
# deadline=1
# noop=2
#
# disk_type
# sata=0
# sas=1
# 500sas=2
# ssd=3
#

import ModelMapExperiment
import ModelMapMap

from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, SVR
from PolynomialRegressor import PolynomialRegressor
from DatabaseRegressor import DatabaseRegressor
import GPy
from GPyRegressor import GPyRegressor, GPyRegressorMultiTask

from Chorus3.ChorusRegressor import ChorusIncrementalRegressor

import logging
import numpy as np
from sklearn.model_selection import GridSearchCV

# create logger with 'modelmap'
logger = logging.getLogger("ModelMap")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
file_logger = logging.FileHandler('modelmap.log')

# create console handler with a higher log level
console_logger = logging.StreamHandler()

# ----------------------------------------------------------------------------
file_logger.setLevel(logging.DEBUG)
console_logger.setLevel(logging.DEBUG)
# ----------------------------------------------------------------------------

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_logger.setFormatter(formatter)
console_logger.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(file_logger)
logger.addHandler(console_logger)

svr_param_grid = {
    "C": np.logspace(-8, 8, 16),
    "gamma": np.logspace(-8, 8, 16)
}

svr_map_grid_search = GridSearchCV(SVR(kernel='rbf', epsilon=0.0001, tol=0.001, shrinking=False),
                                   param_grid=svr_param_grid)
svr_map_grid_search.label = "Linear SVR grid search"

svr_direct_grid_search = GridSearchCV(SVR(kernel='rbf', epsilon=0.0001, tol=0.001, shrinking=False),
                                 param_grid=svr_param_grid)

######################################################################################################
# Chorus regressor

fit_dimension = 0

# Chorus G_INV model
G_INV_config = {"fit_column": fit_dimension + 1}

# Chorus G_RGN model
G_RGN_config = {"fit_column": fit_dimension + 1,
                "type": "poly",
                "order": "4"
                }

# Chorus B_SVM global model
B_SVM_config = {"USE_AVG": "false"}

# Chorus B_LIN global region split model
B_LIN_config = {"max_regions": 2,
                "specdb": {0: (-1.6, 1.6, 0),
                           1: (-1.6, 1.6, 0),
                           2: (-1.6, 1.6, 0)
                           }
                }
# Full Chorus3 ensemble model
Chorus_config = {
    "models": {
        "Chorus_G_RGN": G_RGN_config,
        "Chorus_B_CNST": dict(),
        "Chorus_B_LIN": B_LIN_config,
        "Chorus_B_SVM": B_SVM_config,
    },
    "max_regions": 2,
    "n_way_cross": 3,
    "specdb": {0: (-1.6, 1.6, 0),
               1: (-1.6, 1.6, 0),
               2: (-1.6, 1.6, 0)
               }
}

######################################################################################################

seed = 1

NUM_RUNS = 30

LIN = type('LIN', LinearRegression.__bases__, dict(LinearRegression.__dict__))
LIN.label = "Linear regression"

POL = type('POL', PolynomialRegressor.__bases__, dict(PolynomialRegressor.__dict__))
POL.label = "Polynomial regression"

LINSVR = type('POL', LinearSVR.__bases__, dict(LinearSVR.__dict__))
LINSVR.label = "Linear SVR"

GPyR = type('GPyR', GPyRegressor.__bases__, dict(GPyRegressor.__dict__))
GPyR.label = "Gaussian Process"

GPyMT = type('GPyMT', GPyRegressorMultiTask.__bases__, dict(GPyRegressorMultiTask.__dict__))
GPyMT.label = "Multi-Task Gaussian Process*"

CHOR = ChorusIncrementalRegressor
CHOR.label = "Chorus*"

######################################################################################################

# Authoritative model
auth_model = DatabaseRegressor()

# Dataset files
authoritative_dataset_filename = "Datasets/machine6-7_wrkld1-3-4_ext3_avg_f.csv"
unknown_dataset_filename = "Datasets/machine6-7_wrkld1-3-4_ext4_avg_f.csv"

# Columns containing the function domain
domain_columns = ["workload_id", "machine_id", "io_scheduler", "block_size", "disk_type", "bg_count", "inode_size"]

# Columns containing the function codomain
codomain_columns = "throughput"

# Number of runs to perform
num_runs = NUM_RUNS

# list of Number of samples
sampling_budgets = [5, 10]

# Modeling method to use to create the authoritative model
authoritative_models = auth_model

auth_model = DatabaseRegressor()

# Authoritative model
authoritative_model_dataset_range = {
     "workload_id": [4, 4],
     "machine_id": [6, 6],
     "io_scheduler": [0, 2],
     "disk_type": [0, 3],
     "block_size": [2048, 4096],
     "bg_count": [2, 32],
     "inode_size": [128, 2048]
    }

# Unknown model
unknown_function_dataset_range = {
     "workload_id": [4, 4],
     "machine_id": [6, 6],
     "io_scheduler": [0, 2],
     "disk_type": [0, 3],
     "block_size": [2048, 4096],
     "bg_count": [2, 32],
     "inode_size": [128, 2048]
    }

log2_columns = ["inode_size", "bg_count"]

map_models = [
                LIN(),
                POL(degree=4),
                LINSVR(random_state=seed),
                GPyR(GPy.kern.Matern52(input_dim=5, variance=1.0, lengthscale=1.0)),
            ]

direct_models = [
                LIN(),
                POL(degree=4),
                LINSVR(random_state=seed),
                GPyR(GPy.kern.Matern52(input_dim=7, variance=1.0, lengthscale=1.0)),
                GPyMT(GPy.kern.Matern52(input_dim=7, variance=1.0, lengthscale=1.0)),
                CHOR(Chorus_config),
            ]

fmap = ModelMapMap.ModelMapMapCompositionL_1(auth_model, map_models, ["inode_size", "bg_count", "disk_type", "io_scheduler"])

# -----------------------------------------------------------------------------------------------------------------
#
# Filesystem Scenario 5 - Modeling the effects changing filesystem (ext3 -> ext4)
#
# -----------------------------------------------------------------------------------------------------------------

Filesystem_Experiment_5 = ModelMapExperiment.ModelMapExperiment(
    authoritative_dataset_filename=authoritative_dataset_filename,
    unknown_dataset_filename=unknown_dataset_filename,
    domain_columns=domain_columns,
    codomain_columns=codomain_columns,
    num_runs=num_runs,
    sampling_budgets=sampling_budgets,
    authoritative_models=auth_model,
    authoritative_model_dataset_range=authoritative_model_dataset_range,
    unknown_function_dataset_range=unknown_function_dataset_range,
    map=fmap,
    direct_modeling_methods=direct_models,
    active_learning=False,
    model_selection=False,
    log2_columns=log2_columns,
    logger=logger,
    #normalization='N',
    title="FS-Scenario5-"
    )

Filesystem_Experiment_5.run()
