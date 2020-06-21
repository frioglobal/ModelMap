"""
ModelMap

A toolkit for Model Mapping experimentation.

Model Mapping regression model

Authors:
Francesco Iorio <Frio@cs.toronto.edu>
Ali Hashemi
Michael Tao

License: BSD 3 clause
"""


import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.model_selection import GridSearchCV
import Utils
import logging

class ModelMap(BaseEstimator, RegressorMixin):
    """
    Function mapping class.
    In Model Mapping, one (or more) authoritative model(s) exists.
    When fitting a new function, at the same time that a model of the new function is being built using
    direct_modeling_methods, a map of codomain the authoritative model(s) to the codomain of the new function will be
    created.

    Parameters
    ----------
    authoritative_models : list of instances of scikit-learn models which represents the authoritative models the map
                            will be created from. Each model corresponds to a "slice", to use in N:1 maps, e.g. maps of
                            class (L>1,N).

    map : instance of ModelMapMap to be used as the mapping technique.

    direct_modeling_methods : [optional] list of instances of (empty) scikit-learn models which will be used to train
                                models directly using samples of the unknown function, to provide an estimation of
                                convergence for the map via cross-validation.
    """
    def __init__(self, authoritative_models, map, direct_modeling_methods, model_selection = False, logger=None):
        # Initialize the model class.
        self.authoritative_models = authoritative_models
        self.map = map
        self.direct_modeling_methods = direct_modeling_methods
        self.model_selection = model_selection
        self.logger = logger
        if logger is None:
            import logging
            self.logger = logging.getLogger("ModelMap")
            self.logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"authoritative_models": self.authoritative_models,
                "map": self.map,
                "direct_modeling_methods": self.direct_modeling_methods,
                "model_selection": self.model_selection,
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, x, y):
        """
        Fit a map using a base model as authoritative input.

        Parameters
        ----------
        x : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data

        y : numpy array of shape [n_samples, n_targets]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """

        # Compute accuracy for model selection

        if self.model_selection:
            self._loocv(self.map, x, y)
        else:
            for model in self.direct_modeling_methods:
                model._accuracy_r2 = 0.0
                model._accuracy_mse = 0.0
                model._accuracy_evs = 0.0
                model._accuracy_ma = 0.0
                model._accuracy_rmse = 0.0
            for model in self.map.map_modeling_methods:
                model._accuracy_r2 = 0.0
                model._accuracy_mse = 0.0
                model._accuracy_evs = 0.0
                model._accuracy_ma = 0.0
                model._accuracy_rmse = 0.0

        # Train map models
        self.map.fit(x, y)

        # Train direct models
        for model in self.direct_modeling_methods:
            model.fit(x, y)
            if isinstance(model, GridSearchCV) and self.logger.getEffectiveLevel()<=logging.DEBUG:
                self.logger.debug("model best params: %s", model.best_params_)
                for i_p in range(len(model.cv_results_["params"])):
                    if model.cv_results_["rank_test_score"][i_p] == 1:
                        self.logger.debug("best params: %s", model.cv_results_["params"][i_p])

        return self

    def predict_map(self, x):
        """
        Predict model using the map

        Parameters
        ----------
        x : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : list of arrays, shape = (n_samples,)
            Returns list of predicted values.
        """
        # print()
        # print("ModelMap predictMap: x")
        # print(x)

        return self.map.predict(x)

    def predict_direct(self, x):
        """
        Predict model using the direct models

        Parameters
        ----------
        x : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : list of arrays, shape = (n_samples,)
            Returns list of predicted values.
        """

        result = []
        for model in self.direct_modeling_methods:
            prediction = model.predict(x)
            result.append(prediction)

        return result

    def predict(self, x, prediction_src="best"):
        """
        Predicts the function value at x using prediction_src. prediction_src can be:
            - "best": selects the modeling method that has the lease cross-validation error on the training data.
            If the best modeling methods do not have a significant difference in their cross-validation error,
                - the one with the least variance in cross-validation error will be used to provide the prediction.
            - ?
        Args:
            x:
            prediction_src: [optional] "best"

        Returns:
            An array of predicted value(s) at x. Each item in the array corresponds to a dimension of codomain of the
            function.
        """

        return self.map.predict_cc(x)

    def next_sample_location_index(self, X, direct=False):

        if direct:
            ind = self.direct_modeling_methods[0].next_sample_location_index(X)
        else:
            ind = self.map.next_sample_location_index(X)

        return ind

    def _loocv(self, map, Xr, yr):
        from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
        from sklearn import metrics

        for model in map.map_modeling_methods:
            model.ytests = []
            model.ypreds = []

        loo = LeaveOneOut()

        for train_idx, test_idx in loo.split(Xr):
            X_train, X_test = Xr.iloc[train_idx], Xr.iloc[test_idx]  # requires arrays
            y_train, y_test = yr.iloc[train_idx], yr.iloc[test_idx]

            map.fit(X_train, y_train)
            y_pred = map.predict(X_test)

            for pred, model in zip(y_pred, map.map_modeling_methods):
                model.ytests += list(y_test)
                model.ypreds += list(pred)

        for model in map.map_modeling_methods:

            rr = -metrics.r2_score(model.ytests, model.ypreds)
            ms_error = metrics.mean_squared_error(model.ytests, model.ypreds)
            evs = -metrics.explained_variance_score(model.ytests, model.ypreds)
            ma_error = metrics.mean_absolute_error(model.ytests, model.ypreds)
            rmse = Utils.RMSRE(np.array(model.ytests), np.array(model.ypreds))

            model._accuracy_r2 = rr
            model._accuracy_mse = ms_error
            model._accuracy_evs = evs
            model._accuracy_ma = ma_error
            model._accuracy_rmse = rmse

        for model in self.direct_modeling_methods:
            loo = LeaveOneOut()
            ytests = []
            ypreds = []
            X_array = np.array(Xr)  # r stands for 'regression'
            y_array = np.array(yr)
            for train_idx, test_idx in loo.split(Xr):
                X_train, X_test = X_array[train_idx], X_array[test_idx]  # requires arrays
                y_train, y_test = y_array[train_idx], y_array[test_idx]

                model.fit(X=X_train, y=y_train)
                y_pred = model.predict(X_test)

                # there is only one y-test and y-pred per iteration over the loo.split,
                # so to get a proper graph, we append them to respective lists.

                ytests += list(y_test)
                ypreds += list(y_pred)

            rr = -metrics.r2_score(ytests, ypreds)
            ms_error = metrics.mean_squared_error(ytests, ypreds)
            evs = -metrics.explained_variance_score(ytests, ypreds)
            ma_error = metrics.mean_absolute_error(ytests, ypreds)
            rmse = Utils.RMSRE(np.array(ytests), np.array(ypreds))

            model._accuracy_r2 = rr
            model._accuracy_mse = ms_error
            model._accuracy_evs = evs
            model._accuracy_ma = ma_error
            model._accuracy_rmse = rmse