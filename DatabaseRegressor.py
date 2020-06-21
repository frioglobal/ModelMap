"""
ModelMap

A toolkit for Model Mapping experimentation.

Database regression model (raw data, no interpolation)

Authors:
Francesco Iorio <Frio@cs.toronto.edu>
Ali Hashemi
Michael Tao

License: BSD 3 clause
"""


from sklearn.base import RegressorMixin, BaseEstimator
import numpy as np

class DatabaseRegressor(BaseEstimator, RegressorMixin):
    """
    Identity model, returns x
    """
    def __init__(self):
        pass

    def fit(self, x, y):
        self.modelDict = {}

        # Iterate through all data points
        for i in range(0, len(x.index)):

            # Build the dictionary key as a tuple containing all the point's coordinates
            tempkey = ()
            for j in range(0, len(x.columns)):
                tempkey += (x.values[i, j],)

            # Create a dictionary entry at the point's coordinates, containing the performance value
            self.modelDict[tempkey] = y.values[i]

    def predict(self, x):
        # Iterate through all data points
        result = np.zeros(len(x.index))

        for i in range(0, len(x.index)):

            # Build the dictionary key as a tuple containing all the point's coordinates
            tempkey = ()
            for j in range(0, len(x.columns)):
                tempkey += (x.values[i, j],)

            value = self.modelDict[tempkey]
            result[i] = value

        return result
