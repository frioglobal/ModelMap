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

import math

def generateTrainingData(X, y):
    trainingdata = []
    for i in range(0, X.shape[0]):
        tup = (tuple(X.values[i]), y.values[i])
        trainingdata.append(tup)
    return trainingdata


def generateTestingData(X):
    testingdata = []
    for i in range(0, X.shape[0]):
        tup = (tuple(X.values[i]), (0,))
        testingdata.append(tup)
    return testingdata


def extract_key(conf, fit_dim):
    k = []
    for i in range(0, len(conf)):
        if i != fit_dim:
            k.append(conf[i])
        else:
            x = conf[i]
    return str(k), x


def get_train_db(train_data, fit_dim):
    # convert train_data into dict with keys (conf1,conf2,..)
    train_db = dict()
    for d in train_data:
        key, x = extract_key(d[0], fit_dim)
        y = d[1]
        if key in train_db:
            train_db[key].append((x, y))
        else:
            train_db[key] = [(x, y)]  # first element
    return train_db


# get_region_id
# Uniformly divide configuration space into regions
# We divide each dim into k regions using max-min/k as step
# and use value-min/step to compute id for each dim.
# Region is a tuple (id1,id2, ...idN) in each dimension
# c is conf


def get_region_id(spec, c, k ):
    ndim = len(list(spec.keys()))
    # figure out division size in each dimension
    divs = [0]*ndim
    for i in range(0,ndim):
        divs[i] = ( spec[i][1] - spec[i][0] )/k
    # convert config into rid
    rid = [0]*len(divs)
    for i in range(0,ndim):
        if( divs[i] > 0 ):
            rid[i] = int((( c[i] - spec[i][0] ) / divs[i]))
        else:
            rid[i] = 0
        if( rid[i] >= k ):
            rid[i] = k - 1
        rkey=tuple(r for r in rid)
    return rkey

# log a list


def alog(x):
    y = []
    for element in x:
        y.append(math.log(element))
    return y

# exp a list

def aexp(x):
    y = []
    for element in x:
        y.append(math.exp(element))
    return y

#neg a list


def aneg(x):
    y=[]
    for element in x:
        y.append(-element)
    return y

#power a list


def apower(x,a):
    y=[]
    for element in x:
        y.append(math.pow(element,a))
    return y