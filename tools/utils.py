# -*- coding: utf-8 -*-
"""
Created on Mon May 23 01:00:05 2016

@author: George

Various helper functions
"""

import numpy as np
import os
import errno

from sklearn.metrics import mean_squared_error
def get_rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return (mean_squared_error(pred, actual))**(0.5)

def last_word(input, words=1):
    """ Returns the last words(. delimited) from the end of string """
    for i in range(len(input)-1,0,-1):
        # Count spaces in the string.
        if input[i] == '.':
            words -= 1
        if words == 0:
            # Return the slice up to this point.
            return input[i+1:-2]
    return ""

def mse2rmse(mse):
    """ Converts MSE to RMSE checking for negativity in case MSE comes from sklearn grid_search """   
    if np.all(mse < 0):
        return (mse*-1)**0.5
    else:
        return mse**0.5
        
def generateLogRange(base, start, stop):
    """ logspace wrapper """
    return np.logspace(start, stop, num=stop-start+1, base=base)
    
def check_mkdir(filename):
    """ Check if file path exists, and if not creates the path """
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise     

from numpy.lib.stride_tricks import as_strided
def subtract(X,v):
    """ subtracts a vector v only from non-zero elements of sparse matrix X """
    rows, cols = X.shape
    row_start_stop = as_strided(X.indptr, shape=(rows, 2),
                    strides=2*X.indptr.strides)
    for row, (start, stop) in enumerate(row_start_stop):
        data = X.data[start:stop]
        data -= v[row]   

import csv
def write2csv(filename, list_name):
    
    with open(filename,'w') as f:
        writer = csv.writer(f)
        header = [['parameters','mean RMSE','RMSE std','mean MAE','MAE std','mean training time','RMSEs','MAEs']]
        writer.writerows(header)
        writer.writerows(list_name)

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        li = list(reader)        
    return li

    
def write_dict2csv(filename, dict_name):
    with open(filename,'w') as f:
        writer = csv.writer(f)
        for key, value in dict_name.items():
            writer.writerow([key, value])
        
def read_dict_from_csv(filename):
    with open(filename, 'r') as f:    
        reader = csv.reader(f)
        mydict = dict(reader)
        # convert strings to numbers
        for k in mydict.keys():
            mydict[k] = eval(mydict[k])
    return mydict
        

from collections import namedtuple
class CVScoreTuple (namedtuple('CVScoreTuple',
                                ('parameters',
                                 'mean_validation_score',
                                 'rmse_std',
                                 'mean_mae',
                                 'mae_std',
                                 'mean_training_time',
                                 'cv_validation_scores',
                                 'mae_validation_scores'))):
    # A raw namedtuple is very memory efficient as it packs the attributes
    # in a struct to get rid of the __dict__ of attributes in particular it
    # does not copy the string for the keys on each instance.
    # By deriving a namedtuple class just to introduce the __repr__ method we
    # would also reintroduce the __dict__ on the instance. By telling the
    # Python interpreter that this subclass uses static __slots__ instead of
    # dynamic attributes. Furthermore we don't need any additional slot in the
    # subclass so we set __slots__ to the empty tuple.
    __slots__ = ()

    def __repr__(self):
        """Simple custom repr to summarize the main info"""
        return "mean RMSE: {0:.5f}, std: {1:.5f}, mean time: {2:.1f} sec \nmean MAE: {3:.5f}, std: {4:.5f} \nparams: {5}".format(
            self.mean_validation_score,
            np.std(self.cv_validation_scores),
            self.mean_training_time,
            self.mean_mae,
            np.std(self.mae_validation_scores),
            self.parameters)


