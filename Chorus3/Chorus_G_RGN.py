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
import sys

# G-RGN - Curve fitting


class Chorus_G_RGN(BaseEstimator, RegressorMixin):
    def __init__(self, config):
        self.config = config
        pass

    def curve_fit(self, config, x, y):

        type = config['type']

        coefs = []

        if type.find("log") != -1:
            lx = cc.alog(x)
            coefs = np.polynomial.polynomial.polyfit(lx, y, 1)
        elif type.find("poly") != -1:
            order = int(config['order'])
            coefs = np.polynomial.polynomial.polyfit(x, y, order)
        elif type.find("exp") != -1:
            negexp = config['negexp']
            if negexp.lower().find('true') != -1:
                x = cc.aneg(x)
            expx = cc.aexp(x)
            coefs = np.polynomial.polynomial.polyfit(expx, y, 1)
        elif type.find("power") != -1:
            a = int(config['a'])
            px = cc.apower(x, a)
            coefs = np.polynomial.polynomial.polyfit(px, y, 1)
        else:
            print("Error: Unknown type!", type)
            sys.exit(-1)

        return coefs

    def curve_predict(self, config, x, model):
        type = config['type']

        if type.find("log") != -1:
            lx = math.log(x)
            pred = np.polynomial.polynomial.polyval(lx, model)
        elif type.find("poly") != -1:
            pred = np.polynomial.polynomial.polyval(x, model)
        elif type.find("exp") != -1:
            negexp = config['negexp']
            if negexp.lower().find('true') != -1:
                x = -x
            expx = math.exp(x)
            pred = np.polynomial.polynomial.polyval(expx, model)
        elif type.find("power") != -1:
            a = int(config['a'])
            px = math.pow(x, a)
            pred = np.polynomial.polynomial.polyval(px, model)
        else:
            print("Error: Unknown type!", type)
            sys.exit(-1)
        return pred

    # Curve fitting model

    def fit(self, X, y):

        _trainingData = cc.generateTrainingData(X, y)

        fit_dim = int(self.config['fit_column']) - 1
        assert (fit_dim >= 0)

        # convert list to dict format
        train_db = cc.get_train_db(_trainingData, fit_dim)

        # train the model on each same conf except the fit_dim is changing
        self.model_db = dict()
        for key in train_db:
            data = train_db[key]
            x = []
            y = []
            for d in data:
                x.append(d[0])
                y.append(d[1])

            if len(set(x)) < 2:
                continue;  # too little data
            model = self.curve_fit(self.config, x, y)
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
                pred = self.curve_predict(self.config,float(x),model)
            else:
                pred = float('nan')
            prediction.append(pred)

        return np.array(prediction).ravel()





























    def _fit(self, X, y):
        type = self.config['type']

        x = X[self.dimension, :]

        if type.find("log") != -1:
            lx = np.log(x)
            self.coefs = np.polynomial.polynomial.polyfit(lx, y, 1)
        elif type.find("poly") != -1:
            order = int(self.config['order'])
            self.coefs = np.polynomial.polynomial.polyfit(x, y, order)
        elif type.find("exp") != -1:
            negexp = self.config['negexp']
            if negexp.lower().find('true') != -1:
                x = np.negative(x)
            expx = np.exp(x)
            self.coefs = np.polynomial.polynomial.polyfit(expx, y, 1)
        elif type.find("power") != -1:
            a = int(self.config['a'])
            px = np.power(x, a)
            self.coefs = np.polynomial.polynomial.polyfit(px, y, 1)
        else:
            print("Error: Unknown type!", type)
            sys.exit(-1)

        return self

    def _predict(self, X):

        type = self.config['type']

        x = X[self.dimension, :]

        if type.find("log") != -1:
            lx = np.log(x)
            pred = np.polynomial.polynomial.polyval(lx, self.coefs)
        elif type.find("poly") != -1:
            pred = np.polynomial.polynomial.polyval(x, self.coefs)
        elif type.find("exp") != -1:
            negexp = self.config['negexp']
            if negexp.lower().find('true') != -1:
                x = np.negative(x)
            expx = np.exp(x)
            pred = np.polynomial.polynomial.polyval(expx, self.coefs)
        elif type.find("power") != -1:
            a = int(self.config['a'])
            px = np.power(x, a)
            pred = np.polynomial.polynomial.polyval(px, self.coefs)
        else:
            print("Error: Unknown type!", type)
            sys.exit(-1)
        return pred
