# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 22:01:06 2016

@author: George
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 23:22:16 2016

@author: George
"""

import os
import clr
import numpy as np

clr.AddReference(os.getcwd() + "\\libs\\MyMediaLite\\MyMediaLite.dll")

from MyMediaLite import IO, RatingPrediction, Eval
from MyMediaLite import Random

from tools import timing as tim

class Base():
    
    def __init__(self):
        Random.set_Seed(1)
        
        
    def fit(self, DATASET, model_name, _):
       
        data_path = ".\\data\\" + DATASET + "\\"
        file_prefix = DATASET + "-f"
        file_train_suffix1 = "-train.csv"
        file_test_suffix = "-test.csv"
        
        fold_rmse = []
        fold_mae = []
        fold_time = []
    
        print self.rec.ToString()
        for cv_index in range(0, 5):
            train_data = IO.RatingData.Read(data_path + file_prefix + str(cv_index + 1) + file_train_suffix1)
            test_data = IO.RatingData.Read(data_path + file_prefix + str(cv_index + 1) + file_test_suffix)
            
            print data_path + file_prefix + str(cv_index + 1) + file_train_suffix1
            
            self.rec.Ratings = train_data
            
            tim.startlog('Training model')
            self.rec.Train()
            fold_time.append(tim.endlog('Done training model'))
            
            score = Eval.Ratings.Evaluate(self.rec, test_data)
                  
            fold_rmse.append(score.get_Item("RMSE"))
            fold_mae.append(score.get_Item("MAE"))
        
        print model_name
        print "Mean RMSE: %.5f +- %.5f" % (np.array(fold_rmse, dtype=np.single).mean(), np.array(fold_rmse, dtype=np.single).std())
        print "Mean MAE: %.5f +- %.5f" % (np.array(fold_mae, dtype=np.single).mean(), np.array(fold_mae, dtype=np.single).std())
        
        return fold_rmse, fold_mae, fold_time, 0, 0, 0
        
class GlobalAverage(Base):
    
    def __init__(self):
        Base.__init__(self)
        
        self.rec = RatingPrediction.GlobalAverage()
        
class UserAverage(Base):
    
    def __init__(self):
        Base.__init__(self)
        
        self.rec = RatingPrediction.UserAverage()
        
class ItemAverage(Base):
    
    def __init__(self):
        Base.__init__(self)
        
        self.rec = RatingPrediction.ItemAverage()
        