"""
ModelMap

A toolkit for Model Mapping experimentation.

Error metrics and other utility functions

Authors:
Francesco Iorio <Frio@cs.toronto.edu>
Ali Hashemi
Michael Tao

License: BSD 3 clause
"""


import numpy
import logging
import math
from sklearn.metrics import mean_squared_error

def RMSRE(y_true, y_pred, rel_tol=1e-10):
    """Returns Root Mean Squared Relative Error (RMSRE), i.e. sqrt((mean(((pred-truth)/truth)^2))
    """
    if numpy.all(numpy.isclose(y_true, y_pred, rel_tol)):
        return 0.0
    else:
        if numpy.count_nonzero(y_true) == 0:
            return numpy.nan
        return numpy.sqrt(numpy.sum(numpy.ma.masked_invalid((y_pred - y_true) / (y_true)) ** 2) /
                          numpy.count_nonzero(y_true) )

def RRMSE(y_true, y_pred):
    """Returns Root Relative Mean Squared Error (RRMSE), i.e.
    """
    return 100.0 * numpy.sqrt( numpy.sum( (y_pred - y_true) ** 2 ) / y_true.size ) / (numpy.sum( y_true ) / y_true.size)

def RMSE(y_true, y_pred):
    """Returns Root Mean Squared Error (RMSE), i.e. sqrt(mean((pred-truth)^2))
    """
    return math.sqrt(mean_squared_error(y_true, y_pred))

def MAPE(y_true, y_pred, rel_tol=1e-10):
    """Returns Absolute Percentage Error (MAPE), i.e. (mean(((pred-truth)/truth)) * 100
    """
    if numpy.all(numpy.isclose(y_true, y_pred, rel_tol)):
        return 0.0
    else:
        if numpy.count_nonzero(y_true) == 0:
            return numpy.nan
        return 100.0 * numpy.sum( numpy.ma.masked_invalid( numpy.abs(y_pred - y_true) / (y_true)) ) / numpy.count_nonzero(y_true)

def NMAE(y_true, y_pred):
    """Returns  (abs(true-pred)/num samples)/mean(true)
    """
    mean_true = numpy.sum( y_true ) / numpy.count_nonzero(y_true)
    return ( numpy.sum( numpy.abs(y_pred - y_true) ) / numpy.count_nonzero(y_true) ) / mean_true


def percentile(n):

    def percentile_(x):
        return numpy.percentile(x, n)

    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def print_summary(summary):
    """
    Args:
        summary is an aggregate on a dataframe. THe aggregate should have a "mean" and "sem". 
    """
    # Print mean +- standard error
    for index, row in summary.iterrows():
        print("{}\t{:.3f}\u00B1{:.3f}".format("\t".join([str(x) for x in row.name]), row["mean"], row["sem"]))


def find_min_errors(summary_df, agg_level_name, ttest_pval_th=0.95):
    """
    Runs a pair-wise Welch's t-test between minimum 'mean' value of each group of agg_level_name.
    For any row whose mean is NOT significanly different than the minimum error in its group,
    is_min will be set to True.   
      
    Args:
        summary_df [In/Out] is an aggregate on a dataframe. THe aggregate should have a "mean", "std" and  "count".
        agg_level_name: a level name in summary_df
        ttest_pval_th: pvalue threshold (default: 0.95) 
    """
    from scipy.stats import ttest_ind_from_stats
    summary_df["is_min"] = False
    
    # Find minimum "mean" for each level and its corresponding std and count  
    min_at_level = summary_df.groupby(agg_level_name)["mean", "std", "count"].transform("min")
    for index, row in summary_df.iterrows():
        t_val, p_val =ttest_ind_from_stats(mean1 = min_at_level.loc[index]["mean"],
                             std1 = min_at_level.loc[index]["std"],
                             nobs1 = min_at_level.loc[index]["count"],
                             mean2 = row["mean"],
                             std2 = row["std"],
                             nobs2 = row["count"],
                             equal_var = False)
        if p_val >= ttest_pval_th:
            summary_df.at[index,"is_min"] = True
        else:
            summary_df.at[index,"is_min"] = False


def save_to_file(file_name, data_str):
    """Saves data_str in file_name. If the first try failed,
    adds timestamp to the filename and retries one more time.  
    """
    import os
    import time
    try:
        file = open(file_name, "w")
        file.write(data_str)       
        file.close()
    except Exception as e:
        filenamepath, file_extension = os.path.splitext(file_name)
        file_name2 = filenamepath + time.strftime("%Y%M%d_%I%M%S")+file_extension
        logging.warning("Cannot save csv file %s saving it as %s", file_name, file_name2)
        try:
            file = open(file_name2, "w")
            file.write(data_str)       
            file.close()
        except Exception as e:
            logging.error("Second attempt to write file %s failed: %s", file_name2, e)
            