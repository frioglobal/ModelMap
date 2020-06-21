"""
ModelMap

A toolkit for Model Mapping experimentation.

Gaussian Process and Multi-Task Gaussian Process regression models

Authors:
Francesco Iorio <Frio@cs.toronto.edu>
Ali Hashemi
Michael Tao

License: BSD 3 clause
"""

from sklearn.base import RegressorMixin, BaseEstimator
import GPy
import numpy as np


class GPyRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, kernel):
        self.kernel = kernel
        self.model = None

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"kernel": self.kernel.copy()}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):

        ax = np.array(X)
        ay = np.array([y]).T

        self.model = GPy.models.GPRegression(ax, ay, self.kernel.copy())
        self.model.optimize(messages=False, max_f_eval=1000)

        return self

    def predict(self, X, return_std=False):

        ax = np.array(X)

        pred = self.model.predict(ax)

        if return_std:
            return pred
        else:
            return pred[0].flatten()

    def next_sample_location_index(self, X):

        pred = self.predict(X, return_std=True)

        ind = np.argmax(pred[1])

        return ind


class GPyRegressorMultiTask(BaseEstimator, RegressorMixin):
    def __init__(self, kernel):
        self.kernel = kernel
        self.authX = None
        self.authY = None
        self.model = None
        self.multi_task = True

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"kernel": self.kernel.copy(), "authX": self.authX, "authY": self.authY}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):

        kernel = self.kernel.copy()**GPy.kern.Coregionalize(input_dim=1, output_dim=2, rank=1)

        ax = np.array(X)

        a_1 = np.full((np.size(self.authX,0),1), 0)
        x1 = np.hstack((self.authX,a_1))

        a_2 = np.full((np.size(ax,0),1), 1)
        x2 = np.hstack((ax,a_2))

        total_x = np.append(x1, x2, axis=0)

        y1 = np.array([self.authY]).T

        y2 = np.array([y]).T

        total_y = np.append(y1, y2, axis=0)

        self.model = GPy.models.GPRegression(total_x, total_y, kernel)
        self.model.optimize(messages=False, max_f_eval=1000)

        return self

    def predict(self, X, task=1, return_std=False):

        ax = np.array(X)

        a = np.full((np.size(ax,0),1), task)
        x = np.hstack((ax,a))

        pred = self.model.predict(x)

        if return_std:
            return pred
        else:
            return pred[0].flatten()

    def set_auth_tasks(self, authX, authY):

        self.authX = authX
        self.authY = authY

    def next_sample_location_index(self, X):

        pred = self.predict(X, return_std=True)

        ind = np.argmax(pred[1])

        return ind
