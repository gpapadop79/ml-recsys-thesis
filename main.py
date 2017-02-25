# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 21:24:05 2016

@author: George
"""

from matplotlib import pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.cross_validation import ShuffleSplit

from sklearn.decomposition import TruncatedSVD

import pandas as pd

import rec_datasets
from tools import utils
from tools import sparse
reload(sparse)

from models import apex

import settings
reload(settings)

#== load settings =============================================================

data_path = settings.data_path
DATASET = settings.dataset
num_folds = settings.num_folds
testset_size = settings.testset_size

no_tsvd = settings.no_tsvd
tsvd_components_users = settings.tsvd_components_users
tsvd_components_items = settings.tsvd_components_items

#==============================================================================

import sys
def mem(limit = 1024.0):

    ''' Prints memory usages of objects larger than 1 Mb '''
    sizes = dict((obj, sys.getsizeof(eval(obj))) for obj in globals().keys()) #locals() for current scope
    d = {k: v for k, v in sizes.items() if v > limit**2}
    a = [len(str(x)) for x in d.keys()]
    if len(a)==0:
        print "The are no objects"
        return
    else:
        width_col = max(a)
    
    memory_sum = 0
    for w in sorted(d, key=d.get, reverse=True):
        print "{0:<{col1}} {1:>.1f} Mb".format(w,d[w]/1024.0**2,col1=width_col)
        memory_sum += d[w]/limit**2
    print '------------------------'
    
    if memory_sum > limit:
        print 'Total %.1f Gb' % (memory_sum/limit)
    else:
        print 'Total %.1f Mb' % memory_sum


if __name__ == "__main__":
    
    ## Load the dataset in pandas dataframe
    data = rec_datasets.load_dataset(data_path, DATASET)
    
    # Create a sparse matrix mapping
    sp_map = sparse.SparseMatrixMapping(data)
    
    sp_map.sparsity_info(data)
    
    if num_folds == 1: # if num_folds = 1 do random hold-out split
        cv = ShuffleSplit(len(data), n_iter=1, test_size=testset_size, random_state=0)
    else:
        cv = KFold(len(data), n_folds=num_folds)

    cv_index = 0
    
    for train_idx, test_idx in cv:
        cv_index = cv_index + 1

        ## Sparse matrix construction
        ## X = users
        Xtrain = sp_map.create_sparse_MxN(data, train_idx)
        Xtest = sp_map.create_sparse_MxN(data, test_idx)
        
        ## Y = items
        Ytrain = sp_map.create_sparse_NxM(data, train_idx)
        Ytest = sp_map.create_sparse_NxM(data, test_idx)
        
        
        if no_tsvd == 1: # do not apply tSVD
            Xtrain_svd = Xtrain.toarray()
            Ytrain_svd = Ytrain.toarray()
            
            tsvd_components_users = 0
            tsvd_components_items = 0
        else:
            print "Computing TruncatedSVD users embedding (%d components)" % tsvd_components_users + "\n"
            svd_users = TruncatedSVD(n_components=tsvd_components_users, random_state=0)
            Xtrain_svd = svd_users.fit_transform(Xtrain)
            
            print "Computing TruncatedSVD items embedding (%d components)" % tsvd_components_items + "\n"
            svd_items = TruncatedSVD(n_components=tsvd_components_items, random_state=0)
            Ytrain_svd = svd_items.fit_transform(Ytrain)
            
            print "users tSVD explained variance ratio sum " + str(svd_users.explained_variance_ratio_.sum())
            print "items tSVD explained variance ratio sum " + str(svd_items.explained_variance_ratio_.sum()) + "\n"
    
            
#            #plots
#            plt.figure()
#            plt.scatter(Xtrain_svd[:,0], Xtrain_svd[:,1])
#            
#            plt.figure()
#            plt.scatter(Ytrain_svd[:,0], Ytrain_svd[:,1])
#            
#            #bokeh plot
#            from bokeh.io import output_file, show
#            from bokeh.charts import Scatter
#            output_file("test.html")
#            
#            a = pd.DataFrame(Xtrain_svd)
#            a.columns = ['a1', 'a2']
#           
#            show(Scatter(a, x='a1', y='a2'))     
        
#        A, X_apex = apex.fit_apex(a[:,0:3], 2, verbose=False)
#        
#        plt.figure()
#        plt.scatter(X_apex[:,0], X_apex[:,1])
        

            
        
