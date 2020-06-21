"""
ModelMap

A toolkit for Model Mapping experimentation.

Polynomial regression model

Authors:
Francesco Iorio <Frio@cs.toronto.edu>
Ali Hashemi
Michael Tao

License: BSD 3 clause
"""


from sklearn.linear_model import Ridge
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class PolynomialRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,degree=3):
        self.degree = degree
        self.model = None

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"degree": self.degree}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        self.model = make_pipeline(PolynomialFeatures(degree=self.degree), Ridge())
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
