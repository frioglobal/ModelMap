"""
ModelMap

A toolkit for Model Mapping experimentation.

Chorus incremental modeling framework re-implementation.

Based on the work published in:
Chen, Jin, et al.
"Chorus: an interactive approach to incremental modeling and validation in clouds."
Proceedings of 24th Annual International Conference on Computer Science and Software Engineering. 2014.

Authors:
Francesco Iorio <Frio@cs.toronto.edu>
Ali Hashemi
Michael Tao

License: BSD 3 clause
"""

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

import numpy as np
import math
import Chorus3.Chorus_common as cc

# B-SVM - Support Vector Machine model


class Chorus_B_SVM(BaseEstimator, RegressorMixin):
    def __init__(self, config):
        self.config = config
        pass

    # Linear regression on log transformed data
    def fit(self, X, y):

        # Pre-processing RAW DATA
        self.use_avg = False;  # default value

        avg = self.config['USE_AVG']

        if avg.lower().find('true') != -1:
            self.use_avg = True;

        _X = X
        _y = y

        if (self.use_avg == True):
            self.xscaler = MinMaxScaler()
            self.xscaler.fit(X)
            _X = self.xscaler.transform(X)
            self.yscaler = MinMaxScaler()
            yv = y.values.reshape(1,-1).T
            self.yscaler.fit(yv)
            _y = self.yscaler.transform(yv).ravel()

        self.model = SVR(kernel='rbf', epsilon=0.0001, tol=0.001, shrinking=False)

        self.model.fit(_X, _y)

        return self

    def predict(self, X):
        if (self.use_avg == True):
            _X = self.xscaler.transform(X)
            y = self.model.predict(X)
            yv = y.reshape(1,-1).T
            ret = self.yscaler.inverse_transform(yv).ravel()
            return ret

        pred = self.model.predict(X).ravel()

        return pred

