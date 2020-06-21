"""
ModelMap

A toolkit for Model Mapping experimentation.

Experiment driver class

Authors:
Francesco Iorio <Frio@cs.toronto.edu>
Ali Hashemi
Michael Tao

License: BSD 3 clause
"""


import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import explained_variance_score
import os, errno
import time
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer

import ModelMap
import Utils
from Utils import percentile
from Utils import print_summary
from Utils import find_min_errors
from Utils import save_to_file

EXP_RESULTS_FILE_EXT = "_exp_results.DataFrame"


def truth_table(ds, range):
    """Returns a subset of dataset ds (a data table) whose columns are are filtered by the range defined "range"

    Args:
        ds: A DataFrame object.
        range: dictionary of columns and ranges that represent the subset of the full dataset
                that represent the authoritative slice. The authoritative slice is used
                to train the authoritative model.
                For example, {"buffer_pool": [32, 256]}

    Returns: A DataFrame object. It has the same number of rows as ds and one boolean column which shows whether
        a row met the criteria defined in "range" or not. For example, ds.loc[truth_table(ds, range)] is a subset of
        ds filtered by range
    """
    boolean = True
    for column in range:
        column_name = column
        result = isinstance(range[column], list)
        if result:
            column_min = range[column][0]
            column_max = range[column][1]
            boolean = boolean & (ds[column_name].between(column_min, column_max, inclusive=True))
        else:
            boolean = boolean & (ds[column_name].isin(range[column]))

    return boolean


def identity(X):
    return X


class Slice:
    def __init__(self, filename, data_range, domain_columns, codomain_columns, log2_columns=None,
                   normalization='N', scaler_domain=None, scaler_codomain=None, logger=None):
        self.filename = filename
        self.data_range = data_range
        self.domain_columns = domain_columns
        self.codomain_columns = codomain_columns
        self.log2_columns = log2_columns
        self.normalization = normalization
        self.logger = logger
        self.scaler_domain = scaler_domain
        self.scaler_codomain = scaler_codomain
        self.dataset = None
        self.data = None

    def load(self):
        if self.filename == None:
            return
        # Read CSV file
        self.logger.info("Loading slice dataset from " + str(self.filename))
        self.dataset = pd.read_csv(self.filename)

        # Filter the dataset to extract samples that belong to the slice
        self.logger.info("Filtering slice")
        self.data_original = self.dataset.loc[truth_table(self.dataset, self.data_range)].copy().reset_index(drop=True)
        self.data = self.data_original.copy()

        self.logger.debug("Slice:")
        self.logger.debug('\n' + str(self.data))

        if self.log2_columns:
            self.logger.debug("Applying log2transform")
            transformer = FunctionTransformer(np.log2, inverse_func=np.exp2).fit(self.data_original[self.log2_columns])
            self.data[self.log2_columns] = transformer.transform(self.data_original[self.log2_columns])

        if self.scaler_domain:
            self.logger.debug("Normalizing domain using external scaler")
            self.data[self.domain_columns] = self.scaler_domain.transform(self.data[self.domain_columns])
        else:
            if self.normalization == 'N':
                self.logger.debug("Normalizing domain using MaxMin")
                from sklearn import preprocessing

                self.scaler_domain = preprocessing.MinMaxScaler().fit(self.data[self.domain_columns])
                self.data[self.domain_columns] = self.scaler_domain.transform(self.data[self.domain_columns])

            if self.normalization == 'S':
                self.logger.debug("Normalizing domain to mean=0, stdev=1")
                from sklearn import preprocessing

                self.scaler_domain = preprocessing.StandardScaler().fit(self.data[self.domain_columns])
                self.data[self.domain_columns] = self.scaler_domain.transform(self.data[self.domain_columns])

        if self.scaler_codomain:
            self.logger.debug("Normalizing codomain using external scaler")
            self.data[self.codomain_columns] = self.scaler_codomain.transform(self.data[[self.codomain_columns]])
        else:
            self.scaler_codomain = FunctionTransformer(identity)

            if self.normalization == 'N':
                self.logger.debug("Normalizing codomain using MaxMin")
                from sklearn import preprocessing

                self.scaler_codomain = preprocessing.MinMaxScaler().fit(self.data[[self.codomain_columns]])
                self.data[self.codomain_columns] = self.scaler_codomain.transform(self.data[[self.codomain_columns]])

            if self.normalization == 'S':
                self.logger.debug("Normalizing codomain to mean=0, stdev=1")
                from sklearn import preprocessing

                self.scaler_codomain = preprocessing.StandardScaler().fit(self.data[[self.codomain_columns]])
                self.data[self.codomain_columns] = self.scaler_codomain.transform(self.data[[self.codomain_columns]])


class ModelMapExperiment:
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
                 normalization='S',
                 active_learning = False,
                 active_learning_direct_source = False,
                 model_selection = False,
                 title="Experiment"):
        """
        This is the modeling driver class. It runs different techniques/methods side by side and provided comparison
        metrics

        Parameters
        ----------
            authoritative_dataset_filename : filename of the (csv) dataset that contains the authoritative dataset (
            or both the authoritative dataset and the unknown dataset)

            unknown_dataset_filename : [optional] filename of the (csv) dataset that contains the unknown dataset

            domain_columns : list of dataset column names representing the function domain

            codomain_columns : list of dataset column names representing the function codomain

            num_runs : number of runs to determine statistically significant performance values for models that use
                        random selection of samples

            authoritative_models : list of instances of (empty) scikit-learn models which represents the authoritative
                                    models that will be trained using the provided dataset. Each model corresponds to a
                                    "slice", to use in N:1 maps, e.g. maps of class (L>1,N).

            authoritative_model_dataset_range: dictionary of columns and ranges that represent the subset of the full
                                                dataset that represent the authoritative slice. The authoritative
                                                slice is used to train the authoritative model.
                                                For example, {"buffer_pool": [32, 256]}

            unknown_function_dataset_range: dictionary of columns and ranges that represent the subset of the full
                                            dataset that represent the unknown slice, which is used as both the
                                            training and test set for the maps and the direct models.

            map : instance of ModelMapMap to be used as the mapping technique.

            direct_modeling_methods : [optional] list of instances of (empty) scikit-learn models which will be used
                                        to train models directly using samples of the unknown function, to provide an
                                        estimation of convergence for the map via cross-validation.

            sampling_budgets: a list of number of samples, each will determine number of samples to feed to the map at 
                         every iteration of the experiment 
            logger: A python logger [optional]. If it is not provided by default it will use debug level logging
            title: A human readble title, which can be used in output filenames
        """
        self.authoritative_dataset_filename = authoritative_dataset_filename
        self.unknown_dataset_filename = unknown_dataset_filename
        self.domain_columns = domain_columns
        self.codomain_columns = codomain_columns
        self.num_runs = num_runs
        self.authoritative_models = authoritative_models
        self.authoritative_model_dataset_range = authoritative_model_dataset_range
        self.unknown_function_dataset_range = unknown_function_dataset_range
        self.map = map
        self.direct_modeling_methods = direct_modeling_methods.copy()
        self.sampling_budgets = sampling_budgets if isinstance(sampling_budgets, list) else [sampling_budgets]
        self.plot_graphs = plot_graphs
        self.logger = logger
        self.log2_columns = log2_columns
        self.normalization = normalization
        self.active_learning = active_learning
        self.active_learning_direct_source = active_learning_direct_source
        self.model_selection = model_selection
        if logger is None:
            import logging
            self.logger = logging.getLogger("ModelMap")
            self.logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        self.title = title
        map.set_logger(logger)

    def _loadData(self):
        """
        Loading and preprocessing data
        """

        self.logger.info("Loading authoritative dataset from " + str(self.authoritative_dataset_filename))

        # Check if we are using more than one slice as authoritative
        if isinstance(self.authoritative_dataset_filename, str):
            self.authoritative_slice = Slice(
                filename=self.authoritative_dataset_filename,
                data_range=self.authoritative_model_dataset_range,
                domain_columns=self.domain_columns,
                codomain_columns=self.codomain_columns,
                log2_columns=self.log2_columns,
                normalization = self.normalization,
                logger=self.logger )

            self.authoritative_slice.load()
        else:
            self.authoritative_slice = []
            for filename in self.authoritative_dataset_filename:
                self.authoritative_slice.append( Slice(
                    filename=filename,
                    data_range=self.authoritative_model_dataset_range,
                    domain_columns=self.domain_columns,
                    codomain_columns=self.codomain_columns,
                    log2_columns=self.log2_columns,
                    normalization=self.normalization,
                    logger=self.logger) )

            for slice in self.authoritative_slice:
                slice.load()

        if self.unknown_dataset_filename == None:
            self.unknown_slice = Slice(
                filename=self.authoritative_dataset_filename,
                data_range=self.unknown_function_dataset_range,
                domain_columns=self.domain_columns,
                codomain_columns=self.codomain_columns,
                log2_columns=self.log2_columns,
                normalization = self.normalization,
                scaler_domain=self.authoritative_slice.scaler_domain,
                logger=self.logger )
        else:
            self.logger.info("Loading unknown dataset from " + str(self.unknown_dataset_filename))
            self.unknown_slice = Slice(
                filename=self.unknown_dataset_filename,
                data_range=self.unknown_function_dataset_range,
                domain_columns=self.domain_columns,
                codomain_columns=self.codomain_columns,
                log2_columns=self.log2_columns,
                normalization = self.normalization,
                scaler_domain=self.authoritative_slice.scaler_domain,
                logger=self.logger )

        self.unknown_slice.load()

        # Fit the authoritative model to the training data
        if isinstance(self.authoritative_models, list):

            self.logger.debug("Creating the model of Auth")
            for index in range(len(self.authoritative_models)):
                _X = self.authoritative_slice[index].data[self.domain_columns]
                _y = self.authoritative_slice[index].data[self.codomain_columns]
                self.authoritative_models[index].fit(_X, _y)
                if isinstance(self.authoritative_models[index], GridSearchCV) and self.logger.getEffectiveLevel() <= logging.DEBUG:
                    self.logger.debug("model best params: %s", self.authoritative_models[index].best_params_)
                    for i_p in range(len(self.authoritative_models[index].cv_results_["params"])):
                        if self.authoritative_models[index].cv_results_["rank_test_score"][i_p] == 1:
                            self.logger.debug("best params (rank 1): %s", self.authoritative_models[index].cv_results_["params"][i_p])

            self.logger.info("Authoritative model initial fit")
            for index in range(len(self.authoritative_models)):
                prediction = self.authoritative_models[index].predict(self.authoritative_slice[index].data[self.domain_columns])
                self.logger.info("Authoritative model error (RMSE): %.3f, stdev:%.3f, RMSRE: %.2f%%",
                                 Utils.RMSE(self.authoritative_slice[index].data[self.codomain_columns], prediction),
                                 explained_variance_score(y_true=self.authoritative_slice[index].data[self.codomain_columns],
                                                          y_pred=prediction) if prediction.ndim == 1 else 0.0,
                                 100 * Utils.RMSRE(self.authoritative_slice[index].data[self.codomain_columns], prediction))
                self.authoritative_slice[index].data["auth_model"] = prediction

        else:

            self.logger.debug("Creating the model of Auth")
            _X = self.authoritative_slice.data[self.domain_columns]
            _y = self.authoritative_slice.data[self.codomain_columns]
            self.authoritative_models.fit(_X, _y)
            if isinstance(self.authoritative_models,
                          GridSearchCV) and self.logger.getEffectiveLevel() <= logging.DEBUG:
                self.logger.debug("model best params: %s", self.authoritative_models.best_params_)
                for i_p in range(len(self.authoritative_models.cv_results_["params"])):
                    if self.authoritative_models.cv_results_["rank_test_score"][i_p] == 1:
                        self.logger.debug("best params (rank 1): %s",
                                          self.authoritative_models.cv_results_["params"][i_p])

            self.logger.info("Authoritative model initial fit")
            prediction = self.authoritative_models.predict(
                self.authoritative_slice.data[self.domain_columns])

            self.logger.info("Authoritative model error (RMSE): %.3f, stdev:%.3f, RMSRE: %.2f%%",
                             Utils.RMSE(self.authoritative_slice.data[self.codomain_columns], prediction),
                             explained_variance_score(
                                 y_true=self.authoritative_slice.data[self.codomain_columns],
                                 y_pred=prediction) if prediction.ndim == 1 else 0.0,
                             100 * Utils.RMSRE(self.authoritative_slice.data[self.codomain_columns],
                                               prediction))
            self.authoritative_slice.data["auth_model"] = prediction


    def run(self):
        """ Runs a ModelMapExperiment

        Returns:
            Results of experiment

        """

        self.logger.info("Running experiment %s", self.title)

        results_dir = "results/" + self.title
        try:
            os.makedirs(results_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # Load all data
        self._loadData()

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

        exp_results = pd.DataFrame(columns=["number_of_samples", "run_id", "modeling_technique", "modeling_method",
                                            "rmse", "rmsre", "rrmse", "mape", "acc_mse", "acc_r2", "acc_evs", "acc_ma",
                                            "acc_rmse"])

        # Run multiple iterations to remove random noise

        for run_i in range(0, self.num_runs):

            if self.active_learning:
                indices = np.arange(self.unknown_slice.data.shape[0])
                np.random.seed(run_i)
                np.random.shuffle(indices)
                indices = indices[0:self.sampling_budgets[0]]

            for num_samples in self.sampling_budgets:

                # Check if the budget is larger than the maximum size of the available dataset

                if num_samples > self.unknown_slice.data.shape[0]:
                    self.logger.debug("requested sampling budget %s is greater than the available sample %s",
                                      num_samples, self.unknown_slice.data.shape[0] )
                    num_samples = self.unknown_slice.data.shape[0]

                self.logger.info("Training using " + str(num_samples) + " samples.")

                # Generate the training set randomly.  Set random_state to be able to replicate results.
                # The training set contains different amounts of samples, but using the same random seed, every training
                # set is guaranteed to be a superset of the previous one

                if self.active_learning == False:
                    indices = np.arange(self.unknown_slice.data.shape[0])
                    np.random.seed(run_i)
                    np.random.shuffle(indices)
                    indices = indices[0:num_samples]

                if isinstance(self.authoritative_models, list):
                    training_set = self.authoritative_slice[-1].data.iloc[indices]
                else:
                    training_set = self.unknown_slice.data.iloc[indices]

                # Select anything not in the training set (by index) and put it in the testing set.
                test_set = self.unknown_slice.data.loc[~self.unknown_slice.data.index.isin(training_set.index)]

                test_set_original = self.unknown_slice.data_original[self.codomain_columns].loc[~self.unknown_slice.data.index.isin(training_set.index)]

                # -------------------------------------------------------------
                # -------------------------------------------------------------
                # -------------------------------------------------------------
                # Mapping technique
                # -------------------------------------------------------------
                # -------------------------------------------------------------
                # -------------------------------------------------------------

                # Construct the map
                modelmap = ModelMap.ModelMap(self.authoritative_models,
                                          self.map,
                                          self.direct_modeling_methods,
                                          model_selection = self.model_selection,
                                          logger=self.logger
                                          )

                # Fit the map to the training data.
                self.logger.debug("Map training, run {}, for {} samples".format(run_i, num_samples))
                modelmap.fit(training_set[self.domain_columns], training_set[self.codomain_columns])

                # Generate our predictions for the test set using the map(s).
                self.logger.debug("Map testing, run {}, for {} samples".format(run_i, num_samples))
                predictions_map = modelmap.predict_map(test_set[self.domain_columns])

                # Restore values to un-normalized status
                if predictions_map:
                    predictions_map_original = self.unknown_slice.scaler_codomain.inverse_transform(predictions_map)

                    for index, prediction in enumerate(predictions_map_original):

                        exp_results.loc[exp_results.shape[0]] = [num_samples,
                                                                 run_i,
                                                                 "Mapping",
                                                                 self.map.map_modeling_methods[index].label,
                                                                 Utils.RMSE(test_set_original, prediction),
                                                                 Utils.RMSRE(test_set_original, prediction),
                                                                 Utils.NMAE(test_set_original, prediction),
                                                                 Utils.MAPE(test_set_original, prediction),
                                                                 self.map.map_modeling_methods[index]._accuracy_mse,
                                                                 self.map.map_modeling_methods[index]._accuracy_r2,
                                                                 self.map.map_modeling_methods[index]._accuracy_evs,
                                                                 self.map.map_modeling_methods[index]._accuracy_ma,
                                                                 self.map.map_modeling_methods[index]._accuracy_rmse
                                                                 ]


                # -------------------------------------------------------------------------------------------------
                # Model selection
                # -------------------------------------------------------------------------------------------------
                if self.model_selection:
                    self.logger.debug("Map model selection testing, run {}, for {} samples".format(run_i, num_samples))
                    prediction_map_cc = [modelmap.predict(test_set[self.domain_columns])]

                    predictions_map_cc_original = self.unknown_slice.scaler_codomain.inverse_transform(
                            prediction_map_cc)

                    for index, prediction in enumerate(predictions_map_cc_original):
                        exp_results.loc[exp_results.shape[0]] = [num_samples,
                                                                 run_i,
                                                                "Mapping",
                                                                "Map model selection",
                                                                 Utils.RMSE(test_set_original, prediction),
                                                                 Utils.RMSRE(test_set_original, prediction),
                                                                 Utils.NMAE(test_set_original, prediction),
                                                                 Utils.MAPE(test_set_original, prediction),
                                                                 100000.,
                                                                 100000.,
                                                                 100000.,
                                                                 100000.,
                                                                 100000.
                                                                 ]

                # -------------------------------------------------------------
                # -------------------------------------------------------------
                # -------------------------------------------------------------
                # Direct modeling
                # -------------------------------------------------------------
                # -------------------------------------------------------------
                # -------------------------------------------------------------

                # Generate our predictions for the test set using direct model(s).

                # If the experiment has multiple authoritative models, regenerate the training set.  Set
                # random_state to be able to replicate results.
                # The training set contains different amounts of samples, but using the same random seed, every training
                # set is guaranteed to be a superset of the previous one
                if isinstance(self.authoritative_models, list):

                    indices = np.arange(self.unknown_slice.data.shape[0])
                    np.random.seed(run_i)
                    np.random.shuffle(indices)
                    indices = indices[0:num_samples]

                    training_set = self.unknown_slice.data.iloc[indices]

                    # Select anything not in the training set (by index) and put it in the testing set.
                    test_set = self.unknown_slice.data.loc[~self.unknown_slice.data.index.isin(training_set.index)]

                    test_set_original = self.unknown_slice.data_original[self.codomain_columns].loc[
                        ~self.unknown_slice.data.index.isin(training_set.index)]

                    # Fit the direct models to the training data.
                    self.logger.debug("Direct model training, run {}, for {} samples".format(run_i, num_samples))
                    modelmap.fit(training_set[self.domain_columns], training_set[self.codomain_columns])

                self.logger.debug("Direct model testing, run {}, for {} samples".format(run_i, num_samples))

                # Generate our predictions for the test set using direct modeling.
                predictions_direct = modelmap.predict_direct(test_set[self.domain_columns])

                if predictions_direct:
                    # Restore values to un-normalized status
                    predictions_direct_original = self.unknown_slice.scaler_codomain.inverse_transform(predictions_direct)

                    # Append the MSE of the current direct model(s) to the repository of MSEs, to later compute mean and
                    # std deviation.

                    for index, prediction in enumerate(predictions_direct_original):
                        exp_results.loc[exp_results.shape[0]] = [num_samples,
                                                                 run_i,
                                                                 "Direct",
                                                                 self.direct_modeling_methods[index].label,
                                                                 Utils.RMSE(test_set_original, prediction),
                                                                 Utils.RMSRE(test_set_original, prediction),
                                                                 Utils.NMAE(test_set_original, prediction),
                                                                 Utils.MAPE(test_set_original, prediction),
                                                                 self.direct_modeling_methods[index]._accuracy_mse,
                                                                 self.direct_modeling_methods[index]._accuracy_r2,
                                                                 self.direct_modeling_methods[index]._accuracy_evs,
                                                                 self.direct_modeling_methods[index]._accuracy_ma,
                                                                 self.direct_modeling_methods[index]._accuracy_rmse
                                                                 ]

                if self.active_learning:
                    ind = modelmap.next_sample_location_index(self.unknown_slice.data[self.domain_columns],
                                                           self.active_learning_direct_source)
                    indices = np.append(indices, ind)

            # End of loop over num_samples
        # End of loop over self.num_runs

        #---------------------------------------------------------------------
        # Summarized the experiments
        #---------------------------------------------------------------------

        exp_results.to_pickle(path=os.path.join(results_dir, self.title + EXP_RESULTS_FILE_EXT))
        exp_results.to_csv(os.path.join(results_dir, self.title + "_all.csv"))

        self.generate_summary(exp_results=exp_results,
                  metric_name="rmse",
                  exp_title=self.title,
                  results_dir=results_dir,
                  logger=self.logger
                  )

        self.generate_plot(exp_results=exp_results,
                  metric_name="rmse",
                  exp_title=self.title,
                  results_dir=results_dir,
                  logger=self.logger
                  )

        self.generate_summary(exp_results=exp_results,
                  metric_name="mape",
                  exp_title=self.title,
                  results_dir=results_dir,
                  logger=self.logger
                  )

        self.generate_plot(exp_results=exp_results,
                  metric_name="mape",
                  exp_title=self.title,
                  results_dir=results_dir,
                  logger=self.logger
                  )


        self.generate_summary(exp_results=exp_results,
                  metric_name="rrmse",
                  exp_title=self.title,
                  results_dir=results_dir,
                  logger=self.logger
                  )


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
        if logger.getEffectiveLevel() <= logging.DEBUG: 
            print("---------------------------------------------------")
            print(" " + exp_title + " :: " + metric_name )
            print("---------------------------------------------------")        
            print_summary(summary)
            print("---------------------------------------------------")
    
        find_min_errors(summary_df=summary, agg_level_name="number_of_samples", ttest_pval_th=0.95)
        
        # Create mean+-stderr for each combination 
        summary["mean_pm_sem"] = summary["mean"].map("{:,.2f}".format) + "$\pm$" + summary["sem"].map("{:,.1f}".format)
        
        # add latex command for the minimum error value(s)
        latex_min_err_cmd = "\minerr"  # e.g. \newcommand{\minerr} [1]{{\bfseries \small \color{magenta}#1}}
        min_idx = summary["is_min"] == True
        summary.at[min_idx, "mean_pm_sem"] = summary[min_idx]["mean_pm_sem"].map(lambda x: latex_min_err_cmd + "{" + str(x) + "}")
    
        # Rename the  column names for the final output (latex)        
        summary.index = summary.index.set_names(['\\textbf{Sampling Budget}', '\\textbf{Modeling Technique}', '\\textbf{Modeling Method}'])

        # Create LaTex table (modeling methods as columns)
        table_label = exp_title + metric_name +"Table"
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
        
        csv_filename = os.path.join(results_dir, exp_title + "_results_"+metric_name+".csv")
        try:
            summary.to_csv(csv_filename)
            logger.debug("summary saved to %s", csv_filename)
        except Exception as e:
            csv_filename2 = os.path.join(results_dir, exp_title + "_results_"+metric_name+"_") + time.strftime("%Y%M%d_%I%M%S") + ".csv"
            logger.warning("Cannot save csv file %s saving it as %s", csv_filename, csv_filename2)
            summary.to_csv(csv_filename2)

    @staticmethod
    def generate_plot(exp_results, metric_name, exp_title, results_dir, logger=logging.getLogger()):
        """Creates and saves a plot of exp_results as a LaTex table and a csv file.
        The summary is an aggregation of metric_name over "number_of_samples", "modeling_technique", "modeling_method".


        """
        summary = exp_results.groupby(["number_of_samples", "modeling_technique", "modeling_method"]
                                      )[metric_name].agg(["mean", "std", "count", "sem",
                                                          "min", "max",
                                                          percentile(25), percentile(50),
                                                          percentile(75), percentile(95)
                                                          ])

        #---------------------------------------------------------------------
        # Plot the metric
        x_values = summary.index.levels[0]
        modeling_techniques = summary.index.levels[1]
        modeling_methods = summary.index.levels[2]

        plot = plt.figure()
        labels = list()
        for modeling_technique in modeling_techniques:
            for modeling_method in modeling_methods:
                try:
                    mean_data = summary.loc[[(x, modeling_technique, modeling_method) for x in x_values]]["mean"].tolist()
                    error_data = summary.loc[[(x, modeling_technique, modeling_method) for x in x_values]]["sem"].tolist()
                    if not math.isnan(mean_data[0]):
                        label = str(modeling_technique) + "-" + str(modeling_method)
                        labels.append(label)
                        plt.errorbar(x_values, mean_data, yerr=error_data, label=label,
                                                elinewidth =1)
                except Exception as e:
                    pass
        plt.legend(labels)
        plt.xlabel("Number of Samples")
        plt.ylabel(metric_name)
        plot.show()
        axes = plot.gca()
        y_max = axes.get_ylim()[1]
        if  y_max > 10 * np.percentile(list(summary.percentile_95), 75):
            y_max = np.percentile(list(summary.percentile_95), 75)
            if round(y_max, -2) > min(list(summary.percentile_95)):
                y_max = round(y_max, -2)
            if not np.isnan(y_max):
                axes.set_ylim([0.9*min(summary["min"]),y_max])
        pdf_filename = os.path.join(results_dir, exp_title + "_" + metric_name + ".pdf")
        try:
            pp = PdfPages(pdf_filename)
            pp.savefig(plot)
            pp.close()
        except Exception as e:
            pdf_filename2 = os.path.join(results_dir, exp_title + "_" + metric_name + "_" + time.strftime(
                "%Y%M%d_%I%M%S")+".pdf")
            logger.warning("Cannot save pdf file %s saving it as %s", pdf_filename, pdf_filename2)
            pp = PdfPages(pdf_filename2)
            pp.savefig(plot)
            pp.close()
        axes.set_ylim([0.9*min(summary["min"]),y_max])
        plt.yscale('log')

        pdf_filename = os.path.join(results_dir, exp_title + "_" + metric_name +"_ylog.pdf")
        try:
            pp = PdfPages(pdf_filename)
            pp.savefig(plot)
            pp.close()
        except Exception as e:
            pdf_filename2 = os.path.join(results_dir, exp_title + "_" + metric_name + "_ylog_" + time.strftime(
                "%Y%M%d_%I%M%S")+".pdf")
            logger.warning("Cannot save pdf file %s saving it as %s", pdf_filename, pdf_filename2)
            pp = PdfPages(pdf_filename2)
            pp.savefig(plot)
            pp.close()


class ModelMapCompositeExperiment:

    def __init__(self, title, experiments):
        self.title = title
        self.experiments = experiments

    def run(self):
        results_dir = "results/"

        exp_results = pd.DataFrame(columns=["number_of_samples", "run_id", "modeling_technique", "modeling_method",
                                            "rmse", "rmsre", "rrmse", "mape", "acc_mse", "acc_r2", "acc_evs", "acc_ma",
                                            "acc_rmse"])

        for experiment in self.experiments:
            experiment.run()
            df = pd.read_pickle(path=os.path.join(results_dir + experiment.title, experiment.title + EXP_RESULTS_FILE_EXT))
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
