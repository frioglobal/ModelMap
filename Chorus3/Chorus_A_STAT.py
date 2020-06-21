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
import Chorus3.Chorus_common as cc

# A-STAT - Static global pre-calculated/stored analytical model


class Chorus_A_STAT(BaseEstimator, RegressorMixin):
    def __init__(self, config):
        self.config = config

        X = self.config['x']
        y = self.config['y']

        _resultData = cc.generateTrainingData(X, y)

        self.result_db = dict()
        for item in _resultData:
            if item[0] in self.result_db:
                import sys
                sys.exit(-1)
            else:
                self.result_db[item[0]] = item[1]

    # Linear regression on log transformed data
    def fit(self, X, y):
        return self

    def predict(self, X):

        _testingData = cc.generateTestingData(X)

        prediction = []

        for d in _testingData:
            conf = d[0]
            if conf in self.result_db:
                pred = self.result_db[conf]
            else:
                pred = float('nan')
            prediction.append(pred)

        return np.array(prediction).ravel()

