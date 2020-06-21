"""
ModelMap

A toolkit for Model Mapping experimentation.

Map classes

Authors:
Francesco Iorio <Frio@cs.toronto.edu>
Ali Hashemi
Michael Tao

License: BSD 3 clause
"""

from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import logging

class ModelMapMap:
    """
    Function mapping map class.
    The map class contains the base interface of a map of class

    Parameters
    ----------
    authoritative_models : list of instances of scikit-learn models which represents the authoritative models the map
                            will be created from. Each model corresponds to a "slice".

    map_modeling_methods : list of instances of (empty) scikit-learn models which will be used to train and represent
    the map.

    """
    def __init__(self, authoritative_models, map_modeling_methods, logger=None):
        # Initialize the model class.
        self.authoritative_models = authoritative_models
        self.map_modeling_methods = map_modeling_methods.copy()
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

    def set_logger(self,logger):
        self.logger = logger

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def predict_cc(self, x):
        results = self.predict(x)
        models = self.map_modeling_methods
        models_results = zip(models, results)
        ranked_results = sorted(models_results, key=lambda entry: entry[0]._accuracy_ma)
        return ranked_results[0][1]

# ------------------------------------------------------------------------------------------------------------------
# Composition map
# ------------------------------------------------------------------------------------------------------------------


class ModelMapMapComposition0_1(ModelMapMap):
    """
    Function mapping composition map class.
    The map class contains the implementation of a basic composition map of class 0:1, with l=0 features from C and k=1
    feature from M, which implements y = map(auth(x))

    Parameters
    ----------
    authoritative_models : list of instances of scikit-learn models which represents the authoritative models the map
                            will be created from. Each model corresponds to a "slice".

    map_modeling_methods : list of instances of (empty) scikit-learn models which will be used to train and represent
                           the map.
    """

    def __init__(self, authoritative_models, map_modeling_methods):
        super().__init__(authoritative_models, map_modeling_methods)

    def fit(self, x, y):
        """
        Fit a 0:1 composition map.

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

        # Examine the value of the authoritative model at the requested location, which gives the :1 codomain feature
        authoritative_prediction = self.authoritative_models.predict(x)

        # Transform the authoritative model prediction in a column named '_auth',
        map_domain_dataframe = pd.DataFrame({'_auth': authoritative_prediction})

        # Train map models
        for model in self.map_modeling_methods:
            model.fit(map_domain_dataframe, y)
            if isinstance(model, GridSearchCV) and self.logger.getEffectiveLevel()<=logging.DEBUG:
                self.logger.debug("map model best params: %s", model.best_params_)
                for i_p in range(len(model.cv_results_["params"])):
                    if model.cv_results_["rank_test_score"][i_p] == 1:
                        self.logger.debug("best params: %s", model.cv_results_["params"][i_p])

    def predict(self, x):
        """
        Predict model using the 0:1 composition map

        Parameters
        ----------
        x : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : list of arrays, shape = (n_samples,)
            Returns list of predicted values.
        """

        # Obtain the prediction according to the authoritative model
        authoritative_prediction = self.authoritative_models.predict(x)

        # Create the map domain dataframe. This is a 0:1 class map, so only the authoritative prediction is present
        map_domain_dataframe = pd.DataFrame({'_auth': authoritative_prediction})

        result = []
        for model in self.map_modeling_methods:
            prediction = model.predict(map_domain_dataframe)
            result.append(prediction)

        return result

    def next_sample_location_index(self, x):

        # Obtain the prediction according to the authoritative model
        authoritative_prediction = self.authoritative_models.predict(x)

        # Create the map domain dataframe. This is a l:1 class map, so both the authoritative prediction and the
        # features from the C (function domain) are present
        map_domain_dataframe = pd.DataFrame({'_auth': authoritative_prediction})

        pred = self.map_modeling_methods[0].predict(map_domain_dataframe, return_std=True)

        ind = np.argmax(pred[1])

        return ind


class ModelMapMapCompositionL_1(ModelMapMap):
    """
    Function mapping composition map class.
    The map class contains the implementation of a basic composition map of class l:k, with l=1 features from C and k=1
    feature from M, which implements y = map(z, auth(x)), where z is one dimension (column) of x

    Parameters
    ----------
    authoritative_models : list of instances of scikit-learn models which represents the authoritative models the map
                            will be created from. Each model corresponds to a "slice".

    map_modeling_methods : list of instances of (empty) scikit-learn models which will be used to train and represent
                            the map.

    """

    def __init__(self, authoritative_models, map_modeling_methods, C_feature_names):
        super().__init__(authoritative_models, map_modeling_methods)
        self.C_feature_names = C_feature_names

    def fit(self, x, y):
        """
        Fit a l:1 composition map.

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

        # Examine the value of the authoritative model at the requested location, which gives the :1 codomain feature
        authoritative_prediction = self.authoritative_models.predict(x)

        # Transform the authoritative model prediction in a column named '_auth',
        map_domain_dataframe = pd.DataFrame({'_auth': authoritative_prediction})

        C_features = x[self.C_feature_names].reset_index(drop=True)

        # Append the feature from C
        map_domain_dataframe = map_domain_dataframe.join(C_features)

        # Train map models
        for model in self.map_modeling_methods:
            model.fit(map_domain_dataframe, y)
            if isinstance(model, GridSearchCV) and self.logger.getEffectiveLevel()<=logging.DEBUG:
                self.logger.debug("map model best params: %s", model.best_params_)
                for i_p in range(len(model.cv_results_["params"])):
                    if model.cv_results_["rank_test_score"][i_p] == 1:
                        self.logger.debug("best params: %s", model.cv_results_["params"][i_p])

    def predict(self, x):
        """
        Predict model using the l:1 composition map

        Parameters
        ----------
        x : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : list of arrays, shape = (n_samples,)
            Returns list of predicted values.
        """

        # Obtain the prediction according to the authoritative model
        authoritative_prediction = self.authoritative_models.predict(x)

        # Create the map domain dataframe. This is a l:1 class map, so both the authoritative prediction and the
        # features from the C (function domain) are present
        map_domain_dataframe = pd.DataFrame({'_auth': authoritative_prediction})

        C_features = x[self.C_feature_names].reset_index(drop=True)

        # Append the feature from C
        map_domain_dataframe = map_domain_dataframe.join(C_features)

        result = []
        for model in self.map_modeling_methods:
            prediction = model.predict(map_domain_dataframe)
            result.append(prediction)

        return result


# ------------------------------------------------------------------------------------------------------------------
# Difference map
# ------------------------------------------------------------------------------------------------------------------


class ModelMapMapDifference0_1(ModelMapMap):
    """
    Function mapping difference map class.
    The map class contains the implementation of a basic difference map of class 0:1, with l=0 features from C and k=1
    feature from M, which implements y = auth(x) + map(auth(x))

    Parameters
    ----------
    authoritative_models : list of instances of scikit-learn models which represents the authoritative models the map
                            will be created from. Each model corresponds to a "slice".

    map_modeling_methods : list of instances of (empty) scikit-learn models which will be used to train and represent
                           the map.
    """

    def __init__(self, authoritative_models, map_modeling_methods):
        super().__init__(authoritative_models, map_modeling_methods)

    def fit(self, x, y):
        """
        Fit a 0:1 difference map.

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

        # Examine the value of the authoritative model at the requested location, which gives the :1 codomain feature
        authoritative_prediction = self.authoritative_models.predict(x)

        # Transform the authoritative model prediction in a column named '_auth',
        map_domain_dataframe = pd.DataFrame({'_auth': authoritative_prediction})

        # Generate difference values
        diff = y - authoritative_prediction

        # Train map models
        for model in self.map_modeling_methods:
            model.fit(map_domain_dataframe, diff)
            if isinstance(model, GridSearchCV) and self.logger.getEffectiveLevel()<=logging.DEBUG:
                self.logger.debug("map model best params: %s", model.best_params_)
                for i_p in range(len(model.cv_results_["params"])):
                    if model.cv_results_["rank_test_score"][i_p] == 1:
                        self.logger.debug("best params: %s", model.cv_results_["params"][i_p])

    def predict(self, x):
        """
        Predict model using the 0:1 difference map

        Parameters
        ----------
        x : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : list of arrays, shape = (n_samples,)
            Returns list of predicted values.
        """

        # Obtain the prediction according to the authoritative model
        authoritative_prediction = self.authoritative_models.predict(x)

        # Create the map domain dataframe. This is a 0:1 class map, so only the authoritative prediction is present
        map_domain_dataframe = pd.DataFrame({'_auth': authoritative_prediction})

        result = []
        for model in self.map_modeling_methods:
            prediction = model.predict(map_domain_dataframe)
            result.append(prediction + authoritative_prediction)

        return result


class ModelMapMapDifferenceL_1(ModelMapMap):
    """
    Function mapping difference map class.
    The map class contains the implementation of a difference map of class l:k, with l=1 features from C and k=1
    feature from M, which implements y = auth(x) + map(z, auth(x)), where z is one dimension (column) of x

    Parameters
    ----------
    authoritative_models : list of instances of scikit-learn models which represents the authoritative models the map
                            will be created from. Each model corresponds to a "slice".

    map_modeling_methods : list of instances of (empty) scikit-learn models which will be used to train and represent
                            the map.

    """

    def __init__(self, authoritative_models, map_modeling_methods, C_feature_names):
        super().__init__(authoritative_models, map_modeling_methods)
        self.C_feature_names = C_feature_names

    def fit(self, x, y):
        """
        Fit a l:1 difference map.

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

        # Examine the value of the authoritative model at the requested location, which gives the :1 codomain feature
        authoritative_prediction = self.authoritative_models.predict(x)

        # Transform the authoritative model prediction in a column named '_auth',
        map_domain_dataframe = pd.DataFrame({'_auth': authoritative_prediction})

        C_features = x[self.C_feature_names].reset_index(drop=True)

        # Append the feature from C
        map_domain_dataframe = map_domain_dataframe.join(C_features)

        # Generate difference values
        diff = y - authoritative_prediction

        # Train map models
        for model in self.map_modeling_methods:
            model.fit(map_domain_dataframe, diff)
            if isinstance(model, GridSearchCV) and self.logger.getEffectiveLevel()<=logging.DEBUG:
                self.logger.debug("map model best params: %s", model.best_params_)
                for i_p in range(len(model.cv_results_["params"])):
                    if model.cv_results_["rank_test_score"][i_p] == 1:
                        self.logger.debug("best params: %s", model.cv_results_["params"][i_p])

    def predict(self, x):
        """
        Predict model using the l:1 difference map

        Parameters
        ----------
        x : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : list of arrays, shape = (n_samples,)
            Returns list of predicted values.
        """

        # Obtain the prediction according to the authoritative model
        authoritative_prediction = self.authoritative_models.predict(x)

        # Create the map domain dataframe. This is a l:1 class map, so both the authoritative prediction and the
        # features from the C (function domain) are present
        map_domain_dataframe = pd.DataFrame({'_auth': authoritative_prediction})

        C_features = x[self.C_feature_names].reset_index(drop=True)

        # Append the features from C
        map_domain_dataframe = map_domain_dataframe.join(C_features)

        result = []
        for model in self.map_modeling_methods:
            prediction = model.predict(map_domain_dataframe)
            result.append(prediction + authoritative_prediction)

        return result

    def next_sample_location_index(self, x):

        # Obtain the prediction according to the authoritative model
        authoritative_prediction = self.authoritative_models.predict(x)

        # Create the map domain dataframe. This is a l:1 class map, so both the authoritative prediction and the
        # features from the C (function domain) are present
        map_domain_dataframe = pd.DataFrame({'_auth': authoritative_prediction})

        C_features = x[self.C_feature_names].reset_index(drop=True)

        # Append the features from C
        map_domain_dataframe = map_domain_dataframe.join(C_features)

        pred = self.map_modeling_methods[0].predict(map_domain_dataframe, return_std=True)

        ind = np.argmax(pred[1])

        return ind


class ModelMapMapDifferenceL_K(ModelMapMap):
    """
    Function mapping difference map class.
    The map class contains the implementation of a difference map of class l:k, with l=n features from C and k=m
    feature from M, which implements y = auth(x) + map(z, auth(x)), where z is n dimensions (columns) of x

    Parameters
    ----------
    authoritative_models : list of instances of scikit-learn models which represents the authoritative models the map
                            will be created from. Each model corresponds to a "slice".

    map_modeling_methods : list of instances of (empty) scikit-learn models which will be used to train and represent
                           the map.
    """

    def __init__(self, authoritative_models, map_modeling_methods, C_feature_names, auth_latent_variable,
                 unkn_latent_variable):
        super().__init__(authoritative_models, map_modeling_methods)
        self.C_feature_names = C_feature_names
        self.auth_latent_variable = auth_latent_variable
        self.unkn_latent_variable = unkn_latent_variable

    def fit(self, x, y):
        """
        Fit a l:k difference map.

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

        map_domain_dataframe = pd.DataFrame(columns=['__latent', '_auth'])

        # time 0: (latent value at 0, auth prediction 0) -> auth prediction 0
        # time 1: (latent value at 1, auth prediction 0) -> auth prediction 1
        # time n: (latent value at n, auth prediction 0) -> y

        C_features = x[self.C_feature_names].reset_index(drop=True)

        # Base prediction
        base_auth_prediction = self.authoritative_models[0].predict(x)
        base_model_domain_dataframe = pd.DataFrame({'_auth': base_auth_prediction})
        base_model_domain_dataframe['__latent'] = self.auth_latent_variable[0]
        # Append the feature from C
        base_model_domain_dataframe = base_model_domain_dataframe.join(C_features)

        # Base y
        base_y = pd.DataFrame({'_y': np.zeros(len(base_auth_prediction))})

        # Initialize map dataframe
        map_domain_dataframe = base_model_domain_dataframe

        # Initialize map y
        map_y = base_y

        # Intermediate models
        for index in range(1,len(self.authoritative_models)-1):
            mid_auth_prediction = self.authoritative_models[index].predict(x)
            mid_model_domain_dataframe = pd.DataFrame({'_auth': base_auth_prediction})
            mid_model_domain_dataframe['__latent'] = self.auth_latent_variable[index]
            # Append the feature from C
            mid_model_domain_dataframe = mid_model_domain_dataframe.join(C_features)

            # Intermediate y
            mid_y = pd.DataFrame({'_y': mid_auth_prediction - base_auth_prediction})

            # Update dataset
            map_domain_dataframe = map_domain_dataframe.append(mid_model_domain_dataframe)
            map_y = map_y.append(mid_y)


        # Last model
        last_model_domain_dataframe = pd.DataFrame({'_auth': base_auth_prediction})
        last_model_domain_dataframe['__latent'] = self.auth_latent_variable[-1]
        # Append the feature from C
        last_model_domain_dataframe = last_model_domain_dataframe.join(C_features)

        # Last y
        last_y = pd.DataFrame({'_y': y - base_auth_prediction})

        # Finalize dataset
        map_domain_dataframe = map_domain_dataframe.append(last_model_domain_dataframe)
        map_y = map_y.append(last_y)

        # Train map models
        for model in self.map_modeling_methods:
            model.fit(map_domain_dataframe, map_y['_y'])
            if isinstance(model, GridSearchCV) and self.logger.getEffectiveLevel()<=logging.DEBUG:
                self.logger.debug("map model best params: %s", model.best_params_)
                for i_p in range(len(model.cv_results_["params"])):
                    if model.cv_results_["rank_test_score"][i_p] == 1:
                        self.logger.debug("best params: %s", model.cv_results_["params"][i_p])

    def predict(self, x):
        """
        Predict model using the l:k difference map

        Parameters
        ----------
        x : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : list of arrays, shape = (n_samples,)
            Returns list of predicted values.
        """

        # Generate prediction
        auth_prediction = self.authoritative_models[0].predict(x)
        map_domain_dataframe = pd.DataFrame({'_auth': auth_prediction})
        map_domain_dataframe['__latent'] = self.unkn_latent_variable

        C_features = x[self.C_feature_names].reset_index(drop=True)
        map_domain_dataframe = map_domain_dataframe.join(C_features)

        result = []
        for model in self.map_modeling_methods:
            prediction = model.predict(map_domain_dataframe)
            prediction = prediction.flatten()
            prediction = prediction + auth_prediction
            result.append(prediction.flatten())

        return result
