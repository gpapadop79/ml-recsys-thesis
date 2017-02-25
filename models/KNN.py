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

class KNNBase():
    
    def __init__(self,
                 epsilon = 0,
                 grid_search = 0):
        
        Random.set_Seed(1)
        
        self.epsilon = epsilon
        self.grid_search = grid_search
        
        
    def fit(self, DATASET, model_name, _):
       
        data_path = ".\\data\\" + DATASET + "\\"
        results_path = ".\\results\\" + DATASET + "\\" + model_name + "\\"
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
            
            print 'Fold ', (cv_index + 1), 'RMSE:', score.get_Item("RMSE"), '\n'
            
            model_filename = results_path + "models\\" + DATASET+ '_' + model_name + "_best.model" + str(cv_index + 1)
            try:
            #save the trained model
                self.rec.SaveModel(model_filename)
            except:
                print "\nEXCEPTION\n"
                pass
            
            fold_rmse.append(score.get_Item("RMSE"))
            fold_mae.append(score.get_Item("MAE"))
        
        print model_name
        print "Mean RMSE: %.5f +- %.5f" % (np.array(fold_rmse, dtype=np.single).mean(), np.array(fold_rmse, dtype=np.single).std())
        print "Mean MAE: %.5f +- %.5f" % (np.array(fold_mae, dtype=np.single).mean(), np.array(fold_mae, dtype=np.single).std())
        
        return fold_rmse, fold_mae, fold_time, 0, 0, 0
        
class UserKNN(KNNBase):
    
    def __init__(self, 
                 K = 80,
                 Correlation = 0, # BinaryCosine
                 Alpha = 0,
                 WeightedBinary = False,
                 RegU = 10,
                 RegI = 15,
                 grid_search = 0):
        
        KNNBase.__init__(self, grid_search)
        
        self.rec = RatingPrediction.UserKNN()
        
        self.rec.K = K
        self.rec.Correlation = Correlation
        self.rec.RegU = RegU
        self.rec.RegI = RegI
        self.rec.Alpha = Alpha
        self.rec.WeightedBinary = WeightedBinary
        
