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

# B-LIN - Linear model


def make_vector(c, l):
    x = []
    y = 0
    for v in c:
        x.append(v)
    x.append(1)
    y = l
    return (x, y)


def predict(coeff, x):
    p = 0.0
    ndim = len(x)
    for i in range(0, ndim - 1):
        p += coeff[i] * x[i]
    p += coeff[ndim - 1]
    return p


class Chorus_B_LIN(BaseEstimator, RegressorMixin):
    def __init__(self, config):
        self.config = config

    # Linear regression on log transformed data
    def fit(self, X, y):

        _trainingData = cc.generateTrainingData(X, y)

        # build model using training set
        self.train_info = dict()

        self.max_regions = int(self.config['max_regions'])
        self.specdb = self.config['specdb']

        for k in range(self.max_regions, self.max_regions + 1):
            regions = dict()
            fits = dict()
            # organize data into different regions
            for point in _trainingData:
                region_id = cc.get_region_id(self.specdb, point[0], k)
                if (region_id in regions) is False:
                    regions[region_id] = []
                regions[region_id].append((point[0], point[1]))
            # train for each region
            for region_id in regions.keys():
                A = []
                z = []
                for s in regions[region_id]:
                    x, y = make_vector(s[0], s[1])
                    A.append(x)
                    z.append(y)
                ndims = len(self.specdb)
                if len(z) > ndims:
                    coeff, resid, rank, sigma = np.linalg.lstsq(A, z)
                    # print region_id,coeff,resid,rank,sigma
                    fits[region_id] = (coeff, resid, rank, sigma)
                    # else:
                    # print region_id, 'nan', len(z), ndims
                else:
                    if len(z)>0:
                        fits[region_id] = ([float('nan'), z[0]],)
                    else:
                        fits[region_id] = ([float('nan'), float('nan')],)


            # store everything for later
            self.train_info[k] = (regions, fits)

        return self

    def predict(self, X):

        _testingData = cc.generateTestingData(X)

        prediction = []

        for point in _testingData:
            k = self.max_regions
            region_id = cc.get_region_id(self.specdb, point[0], k)

            if ((region_id in self.train_info[k][0]) is False) or ((region_id in self.train_info[k][1]) is False):
                pred = float('nan')
            else:
                coeff = self.train_info[k][1][region_id][0]
                if np.isnan(coeff[0]):
                    if np.isnan(coeff[1]):
                        pred = float('nan')
                    else:
                        pred = coeff[1]
                else:
                    x, y = make_vector(point[0], point[1])
                    pred = predict(coeff, x)

            prediction.append(pred)

        return np.array(prediction).ravel()
