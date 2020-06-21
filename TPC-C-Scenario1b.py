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
# Database experiments
#
# Uses the dataset created for the work published in:
# Iorio, Francesco, et al.
# "Transfer Learning for Cross-Model Regression in Performance Modeling for the Cloud."
# 2019 IEEE International Conference on Cloud Computing Technology and Science (CloudCom). IEEE, 2019.
#

import ModelMapExperiment
import ModelMapMap

from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, SVR
from PolynomialRegressor import PolynomialRegressor
from DatabaseRegressor import DatabaseRegressor

from Chorus3.ChorusRegressor import ChorusIncrementalRegressor

import GPy
from GPyRegressor import GPyRegressor, GPyRegressorMultiTask

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
                # "type" : "exp",
                # "negexp" : "true"
                }

# Chorus B_SVM global model
B_SVM_config = {"USE_AVG": "false"}

# Chorus B_LIN global region split model
B_LIN_config = {"max_regions": 2,
                # "specdb" : { 0: (0.1, 1.0, 0),
                #             1: (1, 1024, 0),
                #             2: (1, 48, 0)}
                "specdb": {0: (-1.6, 1.6, 0),
                           1: (-1.6, 1.6, 0)
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
    # "specdb" : { 0: (0.1, 1.0, 0),
    #             1: (1, 1024, 0),
    #             2: (1, 48, 0)}
    "specdb": {0: (-1.6, 1.6, 0),
               1: (-1.6, 1.6, 0)
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
# Dataset file
authoritative_dataset_filename = "Datasets/tpcc_blade_20170725.csv"
# Columns containing the function domain
domain_columns = ["buffer_pool", "disk_bandwidth_quanta"]
# Columns containing the function codomain
codomain_columns = "TPMC"
# Number of runs to perform
num_runs = NUM_RUNS
# list of Number of samples
sampling_budgets = [5, 10]
# Modeling method to use to create the authoritative model
authoritative_models = auth_model
# Authoritative model
authoritative_model_dataset_range = {
                                        "disk_bandwidth_quanta": [1, 48],
                                        "CPU": [0.5, 0.5],
                                        "buffer_pool": [1, 1024]
                                    }
# Unknown model
unknown_function_dataset_range = {
                                     "disk_bandwidth_quanta": [1, 48],
                                     "CPU": [0.25, 0.25],
                                     "buffer_pool": [1, 1024]
                                 }

log2_columns = ["CPU", "buffer_pool", "disk_bandwidth_quanta"]

map_models = [
                LIN(),
                POL(degree=4),
                LINSVR(random_state=seed),
                GPyR(GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)),
            ]

direct_models = [
                LIN(),
                POL(degree=4),
                LINSVR(random_state=seed),
                GPyR(GPy.kern.Matern52(input_dim=2, variance=1.0, lengthscale=1.0)),
                GPyMT(GPy.kern.Matern52(input_dim=2, variance=1.0, lengthscale=1.0)),
                CHOR(Chorus_config),
            ]

fmap = ModelMapMap.ModelMapMapDifference0_1(auth_model, map_models)

# -----------------------------------------------------------------------------------------------------------------
#
# TPC-C Scenario 1 - Predicting Performance with Increased CPU Resources
#
# -----------------------------------------------------------------------------------------------------------------

experiment_tpcc_1 = ModelMapExperiment.ModelMapExperiment(
    authoritative_dataset_filename = authoritative_dataset_filename,
    domain_columns = domain_columns,
    codomain_columns = codomain_columns,
    num_runs = num_runs,
    sampling_budgets = sampling_budgets,
    authoritative_models = authoritative_models,
    authoritative_model_dataset_range = authoritative_model_dataset_range,
    unknown_function_dataset_range = unknown_function_dataset_range,
    map = fmap,
    direct_modeling_methods=direct_models,
    active_learning=False,
    model_selection=False,
    log2_columns=log2_columns,
    logger=logger,
    title="TPC-C-Scenario1b-"
)

# Run the experiments

experiment_tpcc_1.run()
