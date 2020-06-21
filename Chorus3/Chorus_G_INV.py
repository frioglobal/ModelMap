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
import numpy as np
import math
import Chorus3.Chorus_common as cc

# G-INV - Inverse exponential model

class Chorus_G_INV(BaseEstimator, RegressorMixin):
    def __init__(self, config):
        self.config = config

    def inverse_fit(self, x, y):
        # change it to log scale
        lx = cc.alog(x)
        ly = cc.alog(y)
        # linear fit
        coefs = np.polynomial.polynomial.polyfit(lx, ly, 1)
        return coefs

    # Linear regression on log transformed data
    def fit(self, X, y):

        _trainingData = cc.generateTrainingData(X, y)

        fit_dim = int(self.config['fit_column']) - 1
        assert (fit_dim >= 0)

        # convert list to dict format
        train_db = cc.get_train_db(_trainingData, fit_dim)

        # train the model
        self.model_db = dict()
        for key in train_db:
            data = train_db[key]
            x = []
            y = []
            for d in data:
                x.append(d[0])
                y.append(d[1])
            # print x,y,len(set(x))
            if len(set(x)) < 2:
                continue;  # too little data
            model = self.inverse_fit(x, y)
            self.model_db[key] = model

        return self

    def predict(self, X):

        _testingData = cc.generateTestingData(X)

        fit_dim = int(self.config['fit_column']) - 1
        assert (fit_dim >= 0)

        prediction = []

        for d in _testingData:
            key, x = cc.extract_key(d[0], fit_dim)
            # do prediction
            if key in self.model_db:
                model = self.model_db[key]
                lx = math.log(x)
                pred = np.exp(np.polynomial.polynomial.polyval(lx, model))
            else:
                pred = float('nan')
            prediction.append(pred)

        return np.array(prediction)

