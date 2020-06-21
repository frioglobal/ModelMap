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


# B-CNST - Constant model


class Chorus_B_CNST(BaseEstimator, RegressorMixin):
    def __init__(self, config):
        pass

    # Linear regression on log transformed data
    def fit(self, X, y):

        _trainingData = cc.generateTrainingData(X, y)

        s = []
        for v in y:
            s.append(v)

        self.avg = sum(s) / len(s)

        return self

    def predict(self, X):

        _testingData = cc.generateTestingData(X)

        prediction = []

        for d in _testingData:
            prediction.append(self.avg)

        return np.array(prediction).ravel()

