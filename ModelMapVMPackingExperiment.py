"""
ModelMap

A toolkit for Model Mapping experimentation.

Virtual Machines Packing experiment driver

Authors:
Francesco Iorio <Frio@cs.toronto.edu>
Ali Hashemi
Michael Tao

License: BSD 3 clause
"""


import pandas as pd
import numpy as np
import os, errno
import time
import logging
import random

import ModelMap
from ModelMapExperiment import ModelMapExperiment
from SimulatedAnnealing import SimulatedAnnealing, ObjectiveFunc
from DatabaseRegressor import DatabaseRegressor

from Utils import percentile
from Utils import print_summary
from Utils import find_min_errors
from Utils import save_to_file

EXP_RESULTS_FILE_EXT = "_vm_packing_exp_results"


class ModelMapVMPackingExperiment(ModelMapExperiment):
    """ModelMapExperiment

    Attributes:
        authoritative_model_dataset_range: dictionary of columns and ranges that represent the subset of the full
            dataset that represent the authoritative slice, which is used to train the authoritative model.
    """

    def __init__(self, authoritative_dataset_filename,
                 domain_columns, codomain_columns,
                 num_runs,
                 authoritative_models,
                 authoritative_model_dataset_range,
                 unknown_function_dataset_range,
                 map,
                 direct_modeling_methods,
                 sampling_budgets,
                 unknown_dataset_filename=None,
                 plot_graphs=False,
                 logger=None,
                 log2_columns=None,
                 normalization='N',
                 run_ground_truth_SA=False,
                 title="Experiment"):
        ModelMapExperiment.__init__(self,
                                    authoritative_dataset_filename=authoritative_dataset_filename,
                                    domain_columns=domain_columns,
                                    codomain_columns=codomain_columns,
                                    num_runs=num_runs,
                                    authoritative_models=authoritative_models,
                                    authoritative_model_dataset_range=authoritative_model_dataset_range,
                                    unknown_function_dataset_range=unknown_function_dataset_range,
                                    map=map,
                                    direct_modeling_methods=direct_modeling_methods,
                                    sampling_budgets=sampling_budgets,
                                    unknown_dataset_filename=unknown_dataset_filename,
                                    plot_graphs=plot_graphs,
                                    logger=logger,
                                    log2_columns=log2_columns,
                                    normalization=normalization,
                                    title=title)
        self.run_ground_truth_SA=run_ground_truth_SA
        self.unknown_model = DatabaseRegressor()

        # Check if the domain is log2
        if self.log2_columns != None:
            self.exp2_domain = True
        else:
            self.exp2_domain = False

    # Model function that returns the value stored in the data dictionary
    def ModelMapPredictMap(self, x):

        nx = np.array([float(x[0]), float(x[1]), float(x[2])])

        coordinates = self.unknown_slice.scaler_domain.transform([nx])

        scoordinates = pd.DataFrame(coordinates, columns=self.domain_columns)

        result = self.modelmap.predict_map(scoordinates)

        result_original = self.unknown_slice.scaler_codomain.inverse_transform(result)

        return result_original[self.modeling_method_index]

    # Model function that returns the value stored in the data dictionary
    def ModelMapPredictDirect(self, x):

        nx = np.array([float(x[0]), float(x[1]), float(x[2])])

        coordinates = self.unknown_slice.scaler_domain.transform([nx])

        scoordinates = pd.DataFrame(coordinates, columns=self.domain_columns)

        result = self.modelmap.predict_direct(scoordinates)

        result_original = self.unknown_slice.scaler_codomain.inverse_transform(result)

        return result_original[self.modeling_method_index]

    # Model function that returns the value stored in the data dictionary for the legacy dataset
    def LegacyModel(self, x):

        nx = np.array([float(x[0]), float(x[1]), float(x[2])])

        coordinates = self.unknown_slice.scaler_domain.transform([nx])

        scoordinates = pd.DataFrame(coordinates, columns=self.domain_columns)

        result_auth = [self.authoritative_models.predict(scoordinates)]

        result_auth_original = self.unknown_slice.scaler_codomain.inverse_transform(result_auth)

        return result_auth_original[0]

    # Model function that returns the value stored in the data dictionary for the unknown function
    def UnknownModel(self, x):

        nx = np.array([float(x[0]), float(x[1]), float(x[2])])

        coordinates = self.unknown_slice.scaler_domain.transform([nx])

        scoordinates = pd.DataFrame(coordinates, columns=self.domain_columns)

        result_auth = [self.unknown_model.predict(scoordinates)]

        result_auth_original = self.unknown_slice.scaler_codomain.inverse_transform(result_auth)

        return result_auth_original[0]

    def run(self):
        """ Runs a ModelMapVMPackingExperiment

        Returns:
            Results of experiment

        """
        self.logger.info("Running experiment %s", self.title)

        # Load all data
        self._loadData()

        # Prepare unknown function ground truth model
        _X = self.unknown_slice.data[self.domain_columns]
        _y = self.unknown_slice.data[self.codomain_columns]
        self.unknown_model.fit(_X, _y)

        # Setup models for multitask learning if necessary by passing the dataset for the authoritative slice

        for model in self.map.map_modeling_methods:
            try:
                if model.multi_task:

                    dom = []

                    if hasattr(self.map, "C_feature_names"):
                        dom = self.map.C_feature_names

                    dom = dom + [self.authoritative_slice.codomain_columns]

                    model.set_auth_tasks(self.authoritative_slice.data[dom],
                                         self.authoritative_slice.data[self.authoritative_slice.codomain_columns])
            except AttributeError:
                pass

        for model in self.direct_modeling_methods:
            try:
                if model.multi_task:
                    model.set_auth_tasks(self.authoritative_slice.data[self.authoritative_slice.domain_columns],
                                         self.authoritative_slice.data[self.authoritative_slice.codomain_columns])
            except AttributeError:
                pass

        # Prepare Simulated Annealing parameters

        # Initial position for SA
        initial_conf = [[0.1, 4, 4], [0.1, 4, 4], [0.1, 4, 4], [0.1, 4, 4]]
        domain = [{0: 0.1, 1: 0.25, 2: 0.5, 3: 1.0},
                  {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32, 6: 48},
                  {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32, 6: 64, 7: 128, 8: 256, 9: 512, 10: 1024}]
        constraints = [1.0, 48, 1024]
        models_unk =  [self.UnknownModel, self.UnknownModel, self.UnknownModel, self.UnknownModel]
        models_map = [self.ModelMapPredictMap, self.ModelMapPredictMap, self.ModelMapPredictMap, self.ModelMapPredictMap]
        models_direct = [self.ModelMapPredictDirect, self.ModelMapPredictDirect, self.ModelMapPredictDirect,
                         self.ModelMapPredictDirect]

        SA_num_steps = 250
        SA_num_substeps = 100

        # Run Simulated Annealing for authoritative model to find true best performance
        random.seed(1)

        if self.run_ground_truth_SA:
            print('Starting optimization on ground truth data')
            auth_results = SimulatedAnnealing(initial_conf, domain, models_unk, constraints, num_steps=SA_num_steps,
                                     num_substeps=SA_num_substeps,log_domain=self.exp2_domain)
        else:
            auth_results = [[[0.25, 16, 256], [0.5, 16, 512], [0.1, 8, 128], [0.1, 8, 128]],[732.71266664]]

        print('Ground truth best configuration = ' + str(auth_results[0]))
        print('Ground truth best total throughput = ' + str(auth_results[1]))

        exp_results = pd.DataFrame(
            columns=["number_of_samples", "run_id", "modeling_technique", "modeling_method", "tpmC", "truth",
                     "error", "config", "actuals"])

        for num_samples in self.sampling_budgets:

            self.logger.info("Training using " + str(num_samples) + " samples.")

            if num_samples > self.unknown_slice.data.shape[0]:
                self.logger.debug("requested sampling budget %s is greater than the available sample %s", num_samples,
                                  self.unknown_slice.data.shape[0])
                num_samples = self.unknown_slice.data.shape[0]

            self.logger.info("Training using " + str(num_samples) + " samples.")

            # Run multiple iterations to remove random noise

            for run_i in [self.num_runs]:

                # Generate the training set.  Set random_state to be able to replicate results.
                # The training set contains progressively more samples, but using the same random seed, every training
                # set is guaranteed to be a superset of the previous one
                training_set = self.unknown_slice.data.sample(num_samples, random_state=run_i + 1)

                # -------------------------------------------------------------
                # Mapping technique
                # Construct the map
                self.modelmap = ModelMap.ModelMap(authoritative_models=self.authoritative_models,
                                               map=self.map,
                                               direct_modeling_methods=self.direct_modeling_methods,
                                               model_selection=self.model_selection,
                                               logger=self.logger
                                               )

                # Fit the map to the training data.
                self.modelmap.fit(training_set[self.domain_columns], training_set[self.codomain_columns])

                for index in range(len(self.map.map_modeling_methods)):

                    self.modeling_method_index = index

                    print('Starting optimization on ' +  self.map.map_modeling_methods[index].label)

                    # Run Simulated Annealing for map model
                    random.seed(run_i + 1)
                    results = SimulatedAnnealing(initial_conf, domain, models_map, constraints, num_steps=SA_num_steps,
                                                 num_substeps=SA_num_substeps,log_domain=self.exp2_domain)

                    print()
                    print(str(index) + " **************************************************************************")
                    print(str(index) + " Map model = " + self.map.map_modeling_methods[index].label)
                    print(str(index) + ' Best predicted configuration using map model = ' + str(results[0]))
                    print(str(index) + ' Best predicted total throughput using map model = ' + str(results[1]))

                    # Compute actual results
                    actual_result = ObjectiveFunc(results[0], models_unk, constraints,log_domain=self.exp2_domain)[0]

                    print(str(index) + ' Actual individual throughput = ')
                    actuals = ""
                    for i in range(len(models_unk)):
                        # Get a performance model from the list
                        fun = models_unk[i]
                        if self.exp2_domain:
                            actuals = actuals + str(fun(np.log2(results[0][i])))
                            print(str(index) + " * " + str(results[0][i]) + "=" + str(fun(np.log2(results[0][i]))))
                        else:
                            actuals = actuals + str(fun(results[0][i]))
                            print(str(index) + " * " + str(results[0][i]) + "=" + str(fun(results[0][i])))

                    print(str(index) + ' Actual total throughput = ' + str(actual_result))

                    print(str(index) + ' Ground truth optimal total throughput = ' + str(auth_results[1][0]))

                    error = abs(actual_result - auth_results[1][0])/auth_results[1][0] * 100.0

                    print(str(index) + ' Error = ' + str(error) + "%")

                    exp_results.loc[exp_results.shape[0]] = [num_samples, run_i, "mapping" , self.map.map_modeling_methods[index].label,
                                                             actual_result, auth_results[1][0], error,
                                                             str(results[0]), actuals]

                for index in range(len(self.direct_modeling_methods)):

                    self.modeling_method_index = index

                    # Run Simulated Annealing for direct model
                    random.seed(run_i + 1)
                    results = SimulatedAnnealing(initial_conf, domain, models_direct, constraints, num_steps=SA_num_steps,
                                                 num_substeps=SA_num_substeps,log_domain=self.exp2_domain)

                    print()
                    print(str(index) + " **************************************************************************")
                    print(str(index) + " Direct model = " + self.direct_modeling_methods[index].label)
                    print(str(index) + ' Best predicted configuration using direct model = ' + str(results[0]))
                    print(str(index) + ' Best predicted total throughput using direct model = ' + str(results[1]))

                    # Compute actual results
                    actual_result = ObjectiveFunc(results[0], models_unk, constraints,log_domain=self.exp2_domain)[0]

                    print(str(index) + ' Actual individual throughput = ' + str(actual_result))
                    actuals = ""
                    for i in range(len(models_unk)):
                        # Get a performance model from the list
                        fun = models_unk[i]
                        if self.exp2_domain:
                            actuals = actuals + str(fun(np.log2(results[0][i])))
                            print(str(index) + " * " + str(results[0][i]) + "=" + str(fun(np.log2(results[0][i]))))
                        else:
                            actuals = actuals + str(fun(results[0][i]))
                            print(str(index) + " * " + str(results[0][i]) + "=" + str(fun(results[0][i])))

                    print(str(index) + ' Ground truth optimal total throughput = ' + str(auth_results[1][0]))

                    error = abs(actual_result - auth_results[1][0])/auth_results[1][0]  * 100.0

                    print(str(index) + ' Error = ' + str(error) + "%")

                    exp_results.loc[exp_results.shape[0]] = [num_samples, run_i, "direct",
                                                             self.direct_modeling_methods[index].label,
                                                             actual_result, auth_results[1][0], error,
                                                             str(results[0]), actuals]


                # End of loop over self.num_runs
        # End of loop over num_samples

        results_dir = "results/" + self.title
        try:
            os.makedirs(results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # Dump iteration results
        exp_results.to_pickle(path=os.path.join(results_dir, self.title + EXP_RESULTS_FILE_EXT + str(self.num_runs) +
                                                ".DataFrame"))
        exp_results.to_csv(os.path.join(results_dir, self.title + EXP_RESULTS_FILE_EXT + str(self.num_runs) + ".csv"))


    @staticmethod
    def generate_summary(exp_results, metric_name, exp_title, results_dir, logger=logging.getLogger()):
        """Creates and saves a summary of exp_results as a LaTex table and a csv file.
        The summary is an aggregation of metric_name over "number_of_samples", "modeling_technique", "modeling_method".


        """
        summary = exp_results.groupby(["number_of_samples", "modeling_technique", "modeling_method"]
                                      )[metric_name].agg(["mean", "std", "count", "sem",
                                                          "min", "max",
                                                          percentile(25), percentile(50),
                                                          percentile(75), percentile(95)
                                                          ])
        # Print mean  standard error
        print("---------------------------------------------------")
        print(" " + exp_title + " :: " + metric_name)
        print("---------------------------------------------------")
        print_summary(summary)
        print("---------------------------------------------------")

        find_min_errors(summary_df=summary, agg_level_name="number_of_samples", ttest_pval_th=0.95)

        # Create mean+-stderr for each combination
        summary["mean_pm_sem"] = summary["mean"].map("{:,.2f}".format) + "\%$\pm$" + summary["sem"].map("{:,"
                                                                                                        ".2f}".format) + "\%"

        # add latex command for the minimum error value(s)
        latex_min_err_cmd = "\minerr"  # e.g. \newcommand{\minerr} [1]{{\bfseries \small \color{magenta}#1}}
        min_idx = summary["is_min"] == True
        summary.at[min_idx, "mean_pm_sem"] = summary[min_idx]["mean_pm_sem"].map(
            lambda x: latex_min_err_cmd + "{" + str(x) + "}")

        # Rename the  column names for the final output (latex)
        summary.index = summary.index.set_names(['Sampling Budget', 'Modeling Technique', 'Modeling Method'])

        # Create LaTex table (modeling methods as columns)
        table_label = exp_title + metric_name + "Table"
        latex_table = summary["mean_pm_sem"].unstack(level=-2)
        # Substitute "None" with "N/A"
        latex_table.replace([None], 'N/A', inplace=True)
        # Bold column names
        rn = lambda a: "\\textbf{" + a + "}"
        latex_table = latex_table.rename(rn, axis='columns')
        latex_table = latex_table.to_latex(escape=False)

        latex_table = "\\newenvironment{" + table_label + """} [2]        
    {\\def\\tableCaption{#1}%
    \\def\\tableLabel{#2}%
    \\begin{table} 
    """ + latex_table + """ 
    \\caption{\\label{\\tableLabel}\\tableCaption}

    \\end{table}
    }
    {}
    """
        save_to_file(file_name=os.path.join(results_dir, table_label + ".tex"),
                     data_str=latex_table)

        csv_filename = os.path.join(results_dir, exp_title + "_results_" + metric_name + ".csv")
        try:
            summary.to_csv(csv_filename)
            logger.debug("summary saved to %s", csv_filename)
        except Exception as e:
            csv_filename2 = os.path.join(results_dir, exp_title + "_results_" + metric_name + "_") + time.strftime(
                "%Y%M%d_%I%M%S") + ".csv"
            logger.warning("Cannot save csv file %s saving it as %s", csv_filename, csv_filename2)
            summary.to_csv(csv_filename2)

