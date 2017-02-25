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


class BaseMatrixFactorizer():
    
    def __init__(self, 
#                 NumFactors=5, 
#                 BiasReg = 0.1, 
#                 RegU = 0.1, 
#                 RegI = 0.1,
#                 FrequencyRegularization = False,
#                 LearnRate = 0.05, 
#                 NumIter = 100, 
#                 BoldDriver = False,
                 epsilon = 0,
                 grid_search = 0):
        
        Random.set_Seed(1)
        
#        self.rec = RatingPrediction.BiasedMatrixFactorization()
        
        # model hyperparameters
#        self.rec.NumFactors=NumFactors
#        self.rec.BiasReg = BiasReg
#        self.rec.RegU = RegU
#        self.rec.RegI = RegI
#        self.rec.FrequencyRegularization = FrequencyRegularization
#        self.rec.LearnRate = LearnRate
#        self.rec.NumIter = NumIter
#        self.rec.BoldDriver = BoldDriver
        
#        self.rec.Decay = 0.9
        
        self.epsilon = epsilon
        self.grid_search = grid_search
    
    def model_init(self, model_name):
        # workaround hack for model initialization
        iter_original = self.rec.NumIter
        self.rec.NumIter = 1
        self.rec.Train()
        
        # Initialize model
        if model_name not in ('FactorWiseMF', 'UserItemBaseline'): #!= 'FactorWiseMF':
            self.rec.InitModel()
        self.rec.NumIter = iter_original
        
        
    def fit(self, DATASET, model_name, show_per_folds):
        
        # hard coded for convergence
        if model_name in ('FactorWiseMF', 'UserItemBaseline'): #== 'FactorWiseMF':
            show_per_folds = 1
       
        data_path = ".\\data\\" + DATASET + "\\"
        results_path = ".\\results\\" + DATASET + "\\" + model_name + "\\"
        file_prefix = DATASET + "-f"
        file_train_suffix1 = "-train.csv"
        file_train_suffix2 = "-train_set.csv"
        file_val_suffix = "-val_set.csv"
        file_test_suffix = "-test.csv"
        
        # cross validation
        fold_rmse = []
        fold_mae = []
        fold_time = []
        exceptions_count=0
        print self.rec.ToString()
        results_rmse = [[],[],[],[],[]]
        results_mae = [[],[],[],[],[]]
        train_error = [[],[],[],[],[]]
        
        for cv_index in range(0, 5):
            
            print "Cross validation: Fold", cv_index + 1
            train_data = IO.RatingData.Read(data_path + file_prefix + str(cv_index + 1) + file_train_suffix1)
            train_set_data = IO.RatingData.Read(data_path + file_prefix + str(cv_index + 1) + file_train_suffix2)
            
            if self.grid_search:
                print data_path + file_prefix + str(cv_index + 1) + file_train_suffix2
            else:
                print data_path + file_prefix + str(cv_index + 1) + file_train_suffix1
            
            val_data = IO.RatingData.Read(data_path + file_prefix + str(cv_index + 1) + file_val_suffix)
            test_data = IO.RatingData.Read(data_path + file_prefix + str(cv_index + 1) + file_test_suffix)
            
            if self.grid_search:
                self.rec.Ratings = train_set_data
            else:
                self.rec.Ratings = train_data
        
            self.model_init(model_name)
            

            tim.startlog('Training model')
            
            for iter in range(0, self.rec.NumIter):                
                self.rec.Iterate()
                
                if (iter + 1) % show_per_folds == 0:
                    
                    if self.grid_search:
                        score = Eval.Ratings.Evaluate(self.rec, val_data) 
                        train_error[cv_index].append(Eval.Ratings.Evaluate(self.rec, train_set_data).get_Item("RMSE"))
                        model_filename = results_path + "models\\" + DATASET+ '_' + model_name + ".model" + str(cv_index + 1)
                    else:
                        score = Eval.Ratings.Evaluate(self.rec, test_data)
                        train_error[cv_index].append(Eval.Ratings.Evaluate(self.rec, train_data).get_Item("RMSE"))
                        model_filename = results_path + "models\\" + DATASET+ '_' + model_name + "_best.model" + str(cv_index + 1)
                        
                    current_rmse = score.get_Item("RMSE")
                    if iter == show_per_folds-1: # if we are at the 1st iteration set error to inf so we save model anyway
                        criterion = np.inf
                    else:
                        criterion = min(results_rmse[cv_index])
                    if current_rmse < criterion: #this model is better than the previous better one
                        try:
                            #save the evaluated model if it's better than the previous
                            self.rec.SaveModel(model_filename)
                        except:
                            print "\nEXCEPTION\n"
                            exceptions_count+=1
                            pass
                    
                    results_rmse[cv_index].append(score.get_Item("RMSE"))
                    results_mae[cv_index].append(score.get_Item("MAE"))
                    idx = len(results_rmse[cv_index])-1 #index of last result
                    
                    if results_rmse[cv_index][idx] - results_rmse[cv_index][idx - 1] < 0:
                        dRMSE = '-'
                    else:
                        dRMSE = '+'
                        
                    if model_name in ('FactorWiseMF', 'UserItemBaseline'):
                        print "Iter", iter + 1, "Train RMSE", train_error[cv_index][idx], "RMSE", results_rmse[cv_index][idx], dRMSE
                    else:
                        print "Iter", iter + 1, "Train RMSE", train_error[cv_index][idx], "RMSE", results_rmse[cv_index][idx], "learn rate", self.rec.current_learnrate, dRMSE
                    
                    margin = results_rmse[cv_index][idx] - min(results_rmse[cv_index]) # rmse deviation of latest epoch from minimum rmse
                    if self.epsilon > 0 and (margin > self.epsilon):
                        print "current RMSE:", results_rmse[cv_index][idx], "> min RMSE:", min(results_rmse[cv_index]), "by at least", self.epsilon
                        print "Reached convergence on training/validation data after", iter + 1, "iterations"                     
                        break
                    
                    
            fold_time.append(tim.endlog('Done training model'))
            print "Min RMSE reached", min(results_rmse[cv_index])
            print "Last RMSE deviation from minimum", margin, "\n"
            
            # Load best saved model (assuming that we were saving every time it got better)
            self.rec.LoadModel(model_filename)
            
            # evaluate on test data
            test_score = Eval.Ratings.Evaluate(self.rec, test_data)
            test_rmse = test_score.get_Item("RMSE")
            test_mae = test_score.get_Item("MAE")
            
            fold_rmse.append(test_rmse)
            fold_mae.append(test_mae)
        

        print "Mean RMSE: %.5f +- %.5f" % (np.array(fold_rmse, dtype=np.single).mean(), np.array(fold_rmse, dtype=np.single).std())
        print "Exceptions:", exceptions_count
        
        return fold_rmse, fold_mae, fold_time, results_rmse, results_mae, train_error

class MatrixFactorizer(BaseMatrixFactorizer):
    
    def __init__(self, NumFactors=5,
                 Regularization = 0.015,
                 LearnRate = 0.05, 
                 NumIter = 100, 
                 epsilon = 0,
                 grid_search = 0):
        
        BaseMatrixFactorizer.__init__(self, epsilon, grid_search)
        
        
        self.rec = RatingPrediction.MatrixFactorization()
        
        # model hyperparameters
        self.rec.NumFactors=NumFactors
        self.rec.Regularization = Regularization
        self.rec.LearnRate = LearnRate
        self.rec.NumIter = NumIter
    
    
class BiasedMatrixFactorizer(BaseMatrixFactorizer):
    
    def __init__(self, NumFactors=5, 
                 BiasReg = 0.1, 
                 RegU = 0.1, 
                 RegI = 0.1,
                 FrequencyRegularization = False,
                 LearnRate = 0.05, 
                 NumIter = 100, 
                 BoldDriver = False,
                 epsilon = 0,
                 grid_search = 0):
        
        BaseMatrixFactorizer.__init__(self, epsilon, grid_search)
        
        
        self.rec = RatingPrediction.BiasedMatrixFactorization()
        
        # model hyperparameters
        self.rec.NumFactors=NumFactors
        self.rec.BiasReg = BiasReg
        self.rec.RegU = RegU
        self.rec.RegI = RegI
        self.rec.FrequencyRegularization = FrequencyRegularization
        self.rec.LearnRate = LearnRate
        self.rec.NumIter = NumIter
        self.rec.BoldDriver = BoldDriver
         
         
         
class SVDPlusPlusMF(BaseMatrixFactorizer):
    
    def __init__(self, NumFactors=5, 
                 BiasReg = 0.33, 
                 Regularization = 0.015,
                 FrequencyRegularization = False,
                 LearnRate = 0.01,
                 BiasLearnRate = 0.7,
                 NumIter = 30,
                 epsilon = 0,
                 grid_search = 0):
        
        BaseMatrixFactorizer.__init__(self, epsilon, grid_search)
        
        
        self.rec = RatingPrediction.SVDPlusPlus()
        
        # model hyperparameters
        self.rec.NumFactors=NumFactors
        self.rec.BiasReg = BiasReg
        self.rec.Regularization = Regularization
        self.rec.FrequencyRegularization = FrequencyRegularization
        self.rec.LearnRate = LearnRate
        self.rec.BiasLearnRate = BiasLearnRate
        self.rec.NumIter = NumIter
        
class FactorWiseMF(BaseMatrixFactorizer):
    
    def __init__(self, NumFactors=10, 
                 Shrinkage = 25, 
                 Sensibility = 0.00001,
                 NumIter = 10,
                 RegU = 10,
                 RegI = 15,
                 epsilon = 0,
                 grid_search = 0):
        
        
        BaseMatrixFactorizer.__init__(self, epsilon, grid_search)
        
        
        self.rec = RatingPrediction.FactorWiseMatrixFactorization()
        
        # model hyperparameters
        self.rec.NumFactors=NumFactors
        self.rec.Shrinkage = Shrinkage
        self.rec.Sensibility = Sensibility
        self.rec.NumIter = NumIter
        self.rec.RegU = RegU
        self.rec.RegI = RegI
        
        
class UserItemBaseline(BaseMatrixFactorizer):
    
    def __init__(self, 
                 RegU = 15, 
                 RegI = 10,
                 NumIter = 10,
                 grid_search = 0):
        
        BaseMatrixFactorizer.__init__(self, grid_search)
        
        self.rec = RatingPrediction.UserItemBaseline()
        
        
        # model hyperparameters
        self.rec.RegU = RegU
        self.rec.RegI = RegI
        self.rec.NumIter = NumIter