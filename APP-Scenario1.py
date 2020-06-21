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
# Application performance experiments
#
# Uses the dataset from the work published in:
# Valov, Pavel, et al.
# "Transferring performance prediction models across different hardware platforms."
# Proceedings of the 8th ACM/SPEC on International Conference on Performance Engineering. 2017.
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

LINM = type('LINM', LinearRegression.__bases__, dict(LinearRegression.__dict__))
LINM.label = "Linear model transfer"

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
authoritative_dataset_filename = "Datasets/Valov_system02.csv"

# Columns containing the function domain
domain_columns = [
"NO_DEBLOCK","NO_FAST_PSKIP","NO_MBTREE","NO_MIXED_REFS","NO_WEIGHTB",
"--rc-lookahead-20","--rc-lookahead-40","--rc-lookahead-60","A","B","C"
]

# Columns containing the function codomain
codomain_columns = "PERF"

# Number of runs to perform
num_runs = NUM_RUNS

# list of Number of samples
sampling_budgets = [5, 10]

# Modeling method to use to create the authoritative model
authoritative_models = auth_model

# Authoritative model
authoritative_model_dataset_range = {
"NO_DEBLOCK" : [0, 1],
"NO_FAST_PSKIP" : [0, 1],
"NO_MBTREE" : [0, 1],
"NO_MIXED_REFS" : [0, 1],
"NO_WEIGHTB" : [0, 1],
"--rc-lookahead-20" : [0, 1],
"--rc-lookahead-40" : [0, 1],
"--rc-lookahead-60" : [0, 1],
"A" : [0, 1],
"B" : [0, 1],
"C" : [0, 1],
"worker_id" : [75, 75]
}

# Unknown model
unknown_function_dataset_range = {
    "NO_DEBLOCK": [0, 1],
    "NO_FAST_PSKIP": [0, 1],
    "NO_MBTREE": [0, 1],
    "NO_MIXED_REFS": [0, 1],
    "NO_WEIGHTB": [0, 1],
    "--rc-lookahead-20": [0, 1],
    "--rc-lookahead-40": [0, 1],
    "--rc-lookahead-60": [0, 1],
    "A": [0, 1],
    "B": [0, 1],
    "C": [0, 1],
    "worker_id": [81, 81]
}

# -----------------------------------------------------------------------------------------------------------------
#
# APP Scenario 1 - Predicting Application Performance with hardware change
# Application: x264
# Configuration: 75 to 81
#
# -----------------------------------------------------------------------------------------------------------------

map_models = [
                LINM(),
            ]

direct_models = []

Application_Experiment_1a = ModelMapExperiment.ModelMapExperiment(
    authoritative_dataset_filename=authoritative_dataset_filename,
    domain_columns=domain_columns,
    codomain_columns=codomain_columns,
    num_runs=num_runs,
    sampling_budgets=sampling_budgets,
    authoritative_models=auth_model,
    authoritative_model_dataset_range=authoritative_model_dataset_range,
    unknown_function_dataset_range=unknown_function_dataset_range,
    map=ModelMapMap.ModelMapMapComposition0_1(auth_model, map_models),
    direct_modeling_methods=direct_models,
    active_learning=False,
    model_selection=False,
    logger=logger,
    title="APP-Scenario1a-"
    )

map_models = [
                LIN(),
                POL(degree=4),
                LINSVR(random_state=seed),
                GPyR(GPy.kern.Matern52(input_dim=2, variance=1.0, lengthscale=1.0)),
            ]

direct_models = [
                LIN(),
                POL(degree=4),
                LINSVR(random_state=seed),
                GPyR(GPy.kern.Matern52(input_dim=11, variance=1.0, lengthscale=1.0)),
                GPyMT(GPy.kern.Matern52(input_dim=11, variance=1.0, lengthscale=1.0)),
                CHOR(Chorus_config),
            ]

fmap = ModelMapMap.ModelMapMapDifferenceL_1(auth_model, map_models, ["NO_MBTREE"])

Application_Experiment_1b = ModelMapExperiment.ModelMapExperiment(
    authoritative_dataset_filename=authoritative_dataset_filename,
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
    logger=logger,
    title="APP-Scenario1b-"
    )

Application_Experiment_1 = ModelMapExperiment.ModelMapCompositeExperiment(
    "APP-Scenario1-",
    [Application_Experiment_1a,
     Application_Experiment_1b,
    ])

Application_Experiment_1.run()