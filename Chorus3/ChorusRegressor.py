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
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import importlib
import Chorus3.Chorus_common as cc

# -------------------------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------------------------


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def compute_rmse_for_one_pred_for_one_meas(pred,meas):
    mse = mean_squared_error(pred, meas)
    rmse = math.sqrt(mse)
    return rmse

def compute_rerr_for_one_pred_for_one_meas(pred,meas):
    rerr_list= []
    for i in range(0,len(pred)):
        if meas[i] == 0.0:
            # Arbitrary large error value when the operation would cause division by zero
            rerr = 1000.0
        else:
            rerr = abs(pred[i]-meas[i])/abs(meas[i])
        rerr_list.append(rerr)
    assert(len(rerr_list)!= 0)
    avg_rerr = sum(rerr_list)/len(rerr_list)
    return avg_rerr


def compute_rerr_ignore_nan(f,y,percent):
    count_nan = 0
    new_f = []
    new_y = []
    for i in range(0,len(f)):
        if math.isnan(f[i]):
            count_nan = count_nan + 1
        else:
            new_f.append(f[i])
            new_y.append(y[i])
    nan_percent = float(count_nan) / float(len(f))
    if nan_percent > percent:
        return float('nan')
    else:
        return compute_rmse_for_one_pred_for_one_meas(new_f,new_y)


def find_neighbor_regions(rid, regions):
    neighbors = []
    dist_list = []
    for r in regions:
        d = 0.0
        assert (len(rid) == len(r))
        for i in range(0, len(r)):
            d += (r[i] - rid[i]) * (r[i] - rid[i])
        d = math.sqrt(d)
        dist_list.append((d, r))
    dist_list.sort()
    for i in range(0, len(dist_list)):
        neighbors.append(dist_list[i][1])
    return neighbors

# -------------------------------------------------------------------------------------
# Chorus ensemble regressor
# -------------------------------------------------------------------------------------


class ChorusRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, config):
        self.config = config

    def fit(self, X, y):

        self.max_regions = int(self.config['max_regions'])
        self.num_folds = int(self.config['n_way_cross'])
        self.specdb = self.config['specdb']

        self.models = []

        # -------------------------------------------------------------------------------------
        # Create internal models from configuration

        for model in sorted(self.config["models"]):
            MyClass = getattr(importlib.import_module("Chorus3."+model), model)
            instance = MyClass(self.config["models"][model])
            self.models.append(instance)

        # -------------------------------------------------------------------------------------
        # Create training and test sets for model scoring by k-fold cross-validation

        mainindex = [i for i in range(X.shape[0])]

        # Init random seed
        rnd = np.random.RandomState(1)

        # Shuffle indices
        rnd.shuffle(mainindex)
        indices = list(split(mainindex, self.num_folds))

        # Create folds for k-fold cross-validation
        folds_train = []
        folds_test = []
        for k in range(0, self.num_folds):
            fold = []
            for j in range(0, self.num_folds):
                if j != k:
                    fold.extend(indices[j])

            folds_train.append((X.iloc[fold], y.iloc[fold]))
            folds_test.append((X.iloc[indices[k]], y.iloc[indices[k]]))

        # -------------------------------------------------------------------------------------
        # Train and test models with k-fold cross-validation
        for model in self.models:
            # Create model regions database
            model.region_db = dict()
            for k in range(0, self.num_folds):
                # Fit model to this fold
                model.fit(folds_train[k][0], folds_train[k][1])
                # Generate predictions for this fold
                pred = model.predict(folds_test[k][0])
                # Assign points to regions
                for i in range(0, folds_test[k][0].shape[0]):
                    # Calculate the region ID for the point's coordinates
                    rid = cc.get_region_id(self.specdb, folds_test[k][0].iloc[i], self.max_regions)
                    # Create the list of points for the current region if not already present
                    if (rid in model.region_db) is False:
                        model.region_db[rid] = []
                    # Create tuple containing coordinates, prediction and ground truth for this point and add to region
                    #                            Coordinates               Prediction    Measurement
                    model.region_db[rid].append((folds_test[k][0].iloc[i], pred[i],      folds_test[k][1].iloc[i]))

        # -------------------------------------------------------------------------------------
        # Calculate cross-validation accuracy score per model, per region
        self.regions = []
        # Iterate through the models and construct accuracy score database
        for model in self.models:
            # Initialize per-region accuracy database for the current model
            model.accdb = dict()
            for rid in model.region_db:
                assert ((rid in model.accdb) is False)
                # Retrieve points assigned to the current region
                data_list = model.region_db[rid]
                pred_list = []
                meas_list = []
                # Construct arrays containing pairs of predictions and ground truth data to calculate error
                for data in data_list:
                    pred_list.append(data[1])  # pred
                    meas_list.append(data[2])  # meas
                assert (len(pred_list) > 0)

                # Calculate relative error between the predictions and ground truth arrays
                # Tolerate a maximum of 10% of nan values
                rerr = compute_rerr_ignore_nan(pred_list, meas_list, 0.20)
                # Store accuracy data for this region
                model.accdb[rid] = rerr

        # Copy regions data from any model, as they should be identical for all models
        self.regions = self.models[0].region_db.keys()

        # -------------------------------------------------------------------------------------
        # Rank models per region
        self.rankdb = dict()
        for rid in self.regions:
            ranks = []
            for m in self.models:
                accdb = m.accdb
                if rid in accdb:
                    accuracy = accdb[rid]
                    ranks.append((accuracy, m))
            sranks = sorted(ranks,key = lambda element: (element[0]))
            self.rankdb[rid] = sranks

        # -------------------------------------------------------------------------------------
        # Fit all models on the entire training data
        for model in self.models:
            model.fit(X, y)

        return self

    def predict(self, X):

        # Perform prediction according to best model in the ensemble
        _testingData = cc.generateTestingData(X)

        prediction = []

        # Iterate through each point to be regressed
        for i in range (0, len(_testingData)):
            k = self.max_regions
            # Find region ID for current point's coordinates
            rid = cc.get_region_id(self.specdb, _testingData[i][0], k)

            # Check if there is a model ranking in the ranking database
            if rid in self.rankdb:
                # Perform prediction using the highest ranked model in the region
                pred = self.predict_rank(X.iloc[[i]], rid)
            else:
                # Find a neighboring region and perform prediction using the highest ranked model in the region
                pred = self.predict_no_rank(X.iloc[[i]], rid)

            # Check the prediction value is acceptable
            upbound = 1000000

            # If predicted value is nan or invalid, fall back to lower-ranked models
            if math.isnan(pred) or pred > upbound: # or pred < 0:

                if rid in self.rankdb:
                    # Iterate through the models in accuracy rank order
                    for m in range(1, len(self.rankdb[rid])):
                        # Get model instance
                        model = self.rankdb[rid][m][1]

                        # Predict using the currently ranked model
                        new_pred = model.predict(X.iloc[[i]])

                        # If the current model's prediction is valid, stop iterating
                        if (math.isnan(new_pred) == False) and new_pred < upbound and new_pred > 0:
                            pred = new_pred
                            break
                else:
                    pred = float('nan')

            # Check if the prediction value is still invalid
            if pred > upbound:
                pred = 0

            # Add prediction to output array (y)
            prediction.append(pred)

        return np.array(prediction).ravel()

    def predict_rank(self, X, rid):
        assert (rid in self.rankdb)
        ranks = self.rankdb[rid]
        top_acc, top_model = ranks[0]
        # find this top model's prediction on conf
        predicted_value = top_model.predict(X)
        return predicted_value

    def predict_no_rank(self, X, rid):
        regions = self.regions
        neighbors = find_neighbor_regions(rid, regions)
        neighbor_rid = neighbors[0]
        pred = self.predict_rank(X, neighbor_rid)
        return pred

class ChorusIncrementalRegressor(ChorusRegressor):
    def __init__(self, config):
        ChorusRegressor.__init__(self, config)
        self.authX = None
        self.authY = None
        self.multi_task = True

    def fit(self, X, y):

        # Use the whole authoritative dataset and replace values with the training set at the corresponding coordinates

        nX = self.authX.copy()
        ny = self.authY.copy()

        ny.update(y)

        return super(ChorusIncrementalRegressor, self).fit(nX, ny)

    def set_auth_tasks(self, authX, authY):

        self.authX = authX
        self.authY = authY
