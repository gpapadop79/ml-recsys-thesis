# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 22:12:31 2016

@author: George
"""

import numpy as np
import scipy.sparse as sp

class SparseMatrixMapping():
    
    def __init__(self, data):
        
        self._uid_users = np.unique(data.iloc[:, 0]) # get unique user IDs
        self._uid_items = np.unique(data.iloc[:, 1]) # get unique item IDs
    
        self.number_of_rows = len(self._uid_users)
        self.number_of_columns = len(self._uid_items)
        
        self.map_users_items()
        
       
    def map_users_items(self):
        
         # =-= user - items indices mapping (in case of non-continuous IDs) ====-===
            self.indices_users = np.zeros((max(self._uid_users),), dtype=np.int)
            self.indices_items = np.zeros((max(self._uid_items),), dtype=np.int)
        
            for i in range(len(self._uid_items)):
                self.indices_items[self._uid_items[i]-1] = i
            for i in range(len(self._uid_users)):
                self.indices_users[self._uid_users[i]-1] = i

    def sparsity_info(self, data):
        ## Print dataset info
        #print "dataset info"
        #print data.describe()
        ## sparsity: percentage of unkwnown ratings of the dataset
        ## sparsity = 1 - density
        sparsity = 1 - len(data)/np.float(self.number_of_columns*self.number_of_rows)
        print "\ndataset sparsity: %.7f or %.2f" % (sparsity, sparsity*100) + '%\n'

    def create_sparse_MxN(self, data, idx):
        
        """
        Creates as csr sparse matrix from data of the form (userid,itemid,rating) 
        using only the indices of idx
        """
        
        return sp.csr_matrix((data.iloc[idx, 2], 
                                    (self.indices_users[data.iloc[idx, 0]-1], 
                                     self.indices_items[data.iloc[idx, 1]-1])), 
                                     shape=(self.number_of_rows, self.number_of_columns))
        
    def create_sparse_NxM(self, data, idx):
        
        """
        Creates as csr sparse matrix from data of the form (userid,itemid,rating) 
        using only the indices of idx
        """
        
        return sp.csr_matrix((data.iloc[idx, 2], 
                                    (self.indices_items[data.iloc[idx, 1]-1], 
                                     self.indices_users[data.iloc[idx, 0]-1])), 
                                     shape=(self.number_of_columns, self.number_of_rows))

