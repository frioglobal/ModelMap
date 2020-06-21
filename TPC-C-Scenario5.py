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

import ModelMapVMPackingExperiment
import ModelMapMap

from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, SVR
from PolynomialRegressor import PolynomialRegressor
from DatabaseRegressor import DatabaseRegressor

import GPy
from GPyRegressor import GPyRegressor, GPyRegressorMultiTask

from Chorus3.ChorusRegressor import ChorusIncrementalRegressor

import pandas as pd
import os
import logging
from multiprocessing import Pool


def experiment_func(iteration):

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

    ######################################################################################################
    # Chorus regressor

    fit_dimension = 0

    # Chorus G_RGN model
    G_RGN_config = {"fit_column": fit_dimension + 1,
                    "type": "poly",
                    "order": "4"
                    }

    # Chorus B_SVM global model
    B_SVM_config = {"USE_AVG": "true"}

    # Chorus B_LIN global region split model
    B_LIN_config = {"max_regions": 2,
                    "specdb": {0: (0.0, 1.0, 0),
                               1: (0.0, 1.0, 0),
                               2: (0.0, 1.0, 0)}
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
        "n_way_cross": 5,
        "specdb": {0: (0.0, 1.0, 0),
                   1: (0.0, 1.0, 0),
                   2: (0.0, 1.0, 0)}
    }

    ######################################################################################################

    seed = 1

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
    unknown_dataset_filename = "Datasets/tpcc_blade_3min_20WH.csv"

    # Columns containing the function domain
    domain_columns = ["CPU", "disk_bandwidth_quanta", "buffer_pool"]

    # Columns containing the function codomain
    codomain_columns = "TPMC"

    # list of Number of samples
    sampling_budgets = [5, 10]

    # Authoritative model
    authoritative_model_dataset_range = {
        "CPU": [0, 1],
        "disk_bandwidth_quanta": [1, 48],
        "buffer_pool": [1, 1024]
    }
    # Unknown model
    unknown_function_dataset_range = {
        "CPU": [0, 1],
        "disk_bandwidth_quanta": [1, 48],
        "buffer_pool": [1, 1024]
    }

    log2_columns = ["CPU", "disk_bandwidth_quanta", "buffer_pool"]

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
        GPyR(GPy.kern.Matern52(input_dim=3, variance=1.0, lengthscale=1.0)),
        GPyMT(GPy.kern.Matern52(input_dim=3, variance=1.0, lengthscale=1.0)),
        CHOR(Chorus_config),
    ]

    fmap = ModelMapMap.ModelMapMapDifferenceL_1(auth_model, map_models, ["disk_bandwidth_quanta"])

    # -----------------------------------------------------------------------------------------------------------------
    #
    # TPC-C Scenario 5 - Modeling VM packing optimization (10WH -> 20WH map class (L>0:1))
    #
    # -----------------------------------------------------------------------------------------------------------------

    experiment = ModelMapVMPackingExperiment.ModelMapVMPackingExperiment(
                                                 authoritative_dataset_filename=authoritative_dataset_filename,
                                                 unknown_dataset_filename=unknown_dataset_filename,
                                                 domain_columns=domain_columns,
                                                 codomain_columns=codomain_columns,
                                                 num_runs=iteration,
                                                 sampling_budgets=sampling_budgets,
                                                 authoritative_models=auth_model,
                                                 authoritative_model_dataset_range=authoritative_model_dataset_range,
                                                 unknown_function_dataset_range=unknown_function_dataset_range,
                                                 map = fmap,
                                                 direct_modeling_methods=direct_models,
                                                 log2_columns=log2_columns,
                                                 logger=logger,
                                                 title="TPC-C-Scenario5-"
                                                 )
    # Run the experiment
    experiment.run()


if __name__ == '__main__':

    num_processes = 12

    # Run Experiment with multiple processes

    iterations = 24
    experiment_title = "TPC-C-Scenario5-"
    EXP_RESULTS_FILE_EXT = "_vm_packing_exp_results"
    results_dir = "results"

    # Invoke experiment on each process

    with Pool(num_processes) as p:
        p.map(experiment_func, range(1,iterations+1))

    # Collect all results

    exp_results = pd.DataFrame(
        columns=["number_of_samples", "run_id", "modeling_technique", "modeling_method", "tpmC", "truth", "error"])

    for i in range(1,iterations+1):
        df = pd.read_pickle(path=os.path.join(results_dir+"/"+experiment_title , experiment_title + EXP_RESULTS_FILE_EXT + str(i) + ".DataFrame"))
        exp_results = exp_results.append(df)

    exp_results.to_csv(os.path.join(results_dir+"/"+experiment_title , experiment_title + EXP_RESULTS_FILE_EXT + ".csv"))

    ModelMapVMPackingExperiment.ModelMapVMPackingExperiment.generate_summary(exp_results=exp_results,
                                                                             metric_name="error",
                                                                             exp_title=experiment_title,
                                                                             results_dir=results_dir+"/"+experiment_title
                                                                             )
