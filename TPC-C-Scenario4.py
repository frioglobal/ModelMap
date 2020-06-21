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

import ModelMapMap
from ModelMapExperiment import ModelMapExperiment, EXP_RESULTS_FILE_EXT

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
import os, errno
import pandas as pd

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

# Dataset file
authoritative_dataset_filename = "Datasets/tpcc_blade_3min_10WH_20170725.csv"
unknown_dataset_filename = "Datasets/tpcc_blade_3min_64WH_20171107.csv"

# Columns containing the function domain
domain_columns = ["CPU", "buffer_pool", "disk_bandwidth_quanta"]

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
                                        "CPU": [0, 1],
                                        "buffer_pool": [1, 1024]
                                    }
# Unknown model
unknown_function_dataset_range = {
                                     "disk_bandwidth_quanta": [1, 48],
                                     "CPU": [0, 1],
                                     "buffer_pool": [1, 1024]
                                 }

log2_columns = ["CPU", "buffer_pool", "disk_bandwidth_quanta"]

direct_models = [
                LIN(),
                POL(degree=4),
                LINSVR(random_state=seed),
                GPyR(GPy.kern.Matern52(input_dim=3, variance=1.0, lengthscale=1.0)),
                GPyMT(GPy.kern.Matern52(input_dim=3, variance=1.0, lengthscale=1.0)),
                CHOR(Chorus_config),
            ]

# -----------------------------------------------------------------------------------------------------------------
#
# TPC-C Scenario 4 - Modeling effects of incremental application performance variation (10WH -> 64WH map class (L>0:1))
#
# -----------------------------------------------------------------------------------------------------------------

map_models_a = [
                LIN(),
                POL(degree=4),
                LINSVR(random_state=seed),
                GPyR(GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)),
            ]

experiment_tpcc_4a = ModelMapExperiment(
    authoritative_dataset_filename = authoritative_dataset_filename,
    unknown_dataset_filename = unknown_dataset_filename,
    domain_columns = domain_columns,
    codomain_columns = codomain_columns,
    num_runs = num_runs,
    sampling_budgets = sampling_budgets,
    authoritative_models = authoritative_models,
    authoritative_model_dataset_range = authoritative_model_dataset_range,
    unknown_function_dataset_range = unknown_function_dataset_range,
    map = ModelMapMap.ModelMapMapComposition0_1(auth_model, map_models_a),
    direct_modeling_methods=direct_models,
    active_learning=False,
    model_selection=False,
    log2_columns=log2_columns,
    normalization='N',
    logger=logger,
    title="TPC-C-Scenario4a-"
)

map_models_b = [
                LIN(),
                POL(degree=4),
                LINSVR(random_state=seed),
                GPyR(GPy.kern.Matern52(input_dim=2, variance=1.0, lengthscale=1.0)),
            ]

experiment_tpcc_4b = ModelMapExperiment(
    authoritative_dataset_filename = authoritative_dataset_filename,
    unknown_dataset_filename = unknown_dataset_filename,
    domain_columns = domain_columns,
    codomain_columns = codomain_columns,
    num_runs = num_runs,
    sampling_budgets = sampling_budgets,
    authoritative_models = authoritative_models,
    authoritative_model_dataset_range = authoritative_model_dataset_range,
    unknown_function_dataset_range = unknown_function_dataset_range,
    map = ModelMapMap.ModelMapMapDifferenceL_1(auth_model, map_models_b, ["disk_bandwidth_quanta"]),
    direct_modeling_methods=direct_models,
    active_learning=False,
    model_selection=False,
    log2_columns=log2_columns,
    normalization='N',
    logger=logger,
    title="TPC-C-Scenario4b-"
)

map_models_c = [
                LIN(),
                POL(degree=4),
                LINSVR(random_state=seed),
                GPyR(GPy.kern.Matern52(input_dim=3, variance=1.0, lengthscale=1.0)),
            ]

experiment_tpcc_4c = ModelMapExperiment(
    authoritative_dataset_filename = authoritative_dataset_filename,
    unknown_dataset_filename = unknown_dataset_filename,
    domain_columns = domain_columns,
    codomain_columns = codomain_columns,
    num_runs = num_runs,
    sampling_budgets = sampling_budgets,
    authoritative_models = authoritative_models,
    authoritative_model_dataset_range = authoritative_model_dataset_range,
    unknown_function_dataset_range = unknown_function_dataset_range,
    map = ModelMapMap.ModelMapMapDifferenceL_1(auth_model, map_models_c, ["CPU", "disk_bandwidth_quanta"]),
    direct_modeling_methods=direct_models,
    active_learning=False,
    model_selection=False,
    log2_columns=log2_columns,
    normalization='N',
    logger=logger,
    title="TPC-C-Scenario4c-"
)

map_models_d = [
                LIN(),
                POL(degree=4),
                LINSVR(random_state=seed),
                GPyR(GPy.kern.Matern52(input_dim=4, variance=1.0, lengthscale=1.0)),
            ]

experiment_tpcc_4d = ModelMapExperiment(
    authoritative_dataset_filename = authoritative_dataset_filename,
    unknown_dataset_filename = unknown_dataset_filename,
    domain_columns = domain_columns,
    codomain_columns = codomain_columns,
    num_runs = num_runs,
    sampling_budgets = sampling_budgets,
    authoritative_models = authoritative_models,
    authoritative_model_dataset_range = authoritative_model_dataset_range,
    unknown_function_dataset_range = unknown_function_dataset_range,
    map = ModelMapMap.ModelMapMapDifferenceL_1(auth_model, map_models_d, ["buffer_pool", "CPU", "disk_bandwidth_quanta"]),
    direct_modeling_methods=direct_models,
    active_learning=False,
    model_selection=False,
    log2_columns=log2_columns,
    normalization='N',
    logger=logger,
    title="TPC-C-Scenario4d-"
)

# Run the experiments

class CompositeExperiment:

    def __init__(self, title, experiments, experiments_mapping_names):
        self.title = title
        self.experiments = experiments
        self.experiments_mapping_names = experiments_mapping_names

    def run(self):
        results_dir = "results/"

        exp_results = pd.DataFrame(columns=["number_of_samples", "run_id", "modeling_technique", "modeling_method",
                                            "rmse", "rmsre", "rrmse", "mape", "acc_mse", "acc_r2", "acc_evs", "acc_ma",
                                            "acc_rmse"])

        for experiment in zip(self.experiments, self.experiments_mapping_names):
            experiment[0].run()
            ex_title = experiment[0].title
            df = pd.read_pickle(path=os.path.join(results_dir + ex_title, ex_title + EXP_RESULTS_FILE_EXT))
            df["modeling_technique"] = df["modeling_technique"].replace('Mapping', experiment[1])
            exp_results = exp_results.append(df)

        results_dir = "results/" + self.title

        try:
            os.makedirs(results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        exp_results.to_csv(os.path.join(results_dir, self.title + EXP_RESULTS_FILE_EXT + ".csv"))

        ModelMapExperiment.generate_summary(exp_results=exp_results,
                                            metric_name="rmse",
                                            exp_title=self.title,
                                            results_dir=results_dir
                                            )

        ModelMapExperiment.generate_summary(exp_results=exp_results,
                                            metric_name="mape",
                                            exp_title=self.title,
                                            results_dir=results_dir
                                            )


experiments = [
    experiment_tpcc_4a,
    experiment_tpcc_4b,
    experiment_tpcc_4c,
    experiment_tpcc_4d,
]

experiments_mapping_names = [
    "Mapping(0,1)",
    "Mapping(1,1)",
    "Mapping(2,1)",
    "Mapping(3,1)",
]

experiment_tpcc_4 = CompositeExperiment(
    "TPC-C-Scenario4-",
    experiments,
    experiments_mapping_names
)

experiment_tpcc_4.run()
