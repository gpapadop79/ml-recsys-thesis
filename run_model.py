# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 23:53:30 2016

@author: George
"""

import sys
import numpy as np
from sklearn.model_selection import ParameterGrid
from tools.utils import CVScoreTuple
from tools.utils import write2csv
from tools.utils import write_dict2csv
from tools.utils import check_mkdir

from models import Baselines
from models import MF
from models import KNN
reload(MF)
reload(KNN)

DATASET = 'ml-100k'
#DATASET = 'ml-1M'
#DATASET = 'jester-1'
#DATASET = 'book-crossing'

#algorithm = 'GlobalAverage'
#algorithm = 'UserAverage'
#algorithm = 'ItemAverage'
#algorithm = 'UserItemBaseline'
algorithm = 'MF'
#algorithm = 'BMF'
#algorithm = 'SVDpp'
#algorithm = 'FactorWiseMF'
#algorithm = 'UserKNN'

# For iterative models
# Every how many folds to eval and show error
show_per_folds = 1
#TODO: put in model parameters 

# Read command line arguments
if len(sys.argv)>1:
    print sys.argv[1]
    print sys.argv[2]
    
    DATASET = str(sys.argv[1])
    algorithm = str(sys.argv[2])

result_dir = '.\\results\\'+ DATASET +'\\' + algorithm + '\\'
log_dir = result_dir + 'logs\\'
models_dir = result_dir + '\\models\\'


# check directories for existence (if not exist -> create)
check_mkdir(result_dir)
check_mkdir(log_dir)
check_mkdir(models_dir)
# ========================================================


if algorithm == 'GlobalAverage' or algorithm == 'UserAverage' or algorithm == 'ItemAverage':
    param_grid = {}
    
elif algorithm == 'UserItemBaseline':
    param_grid = {'NumIter':[20], 'RegU':[0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 50, 100, 1000], 'RegI':[0.01, 0.1, 1, 5, 10, 15, 20, 50, 100], 'grid_search':[1]}
#    param_grid = {'NumIter':[10], 'RegU':[1], 'RegI':[1], 'grid_search':[0]}
    
elif algorithm == 'UserKNN':
    # Correlations
    # 0 = BinaryCosine: binary cosine similarity
    # 1 = Jaccard: Jaccard index (Tanimoto coefficient)
    # 2 = ConditionalProbability: conditional probability
    # 3 = BidirectionalConditionalProbability: bidirectional conditional probability
    # 4 = Cooccurrence: cooccurrence counts
    # 5 = SimilarityProvider: use a similarity provider to get the correlation
    # 6 = Stored: use stored/precomputed correlation
    # 7 = Pearson: Pearson correlation
    # 8 = RatingCosine: rating cosine similarity
		
    param_grid = {'K': [10, 20, 30, 40, 50], 'Correlation':[0, 1, 7], 'grid_search':[0]}
    
    if DATASET == 'ml-100k':
        param_grid.update({'RegU': [5], 'RegI':[1]})
    elif DATASET =='ml-1M':
        param_grid.update({'RegU': [5], 'RegI':[1]})
    elif DATASET == 'jester-1':
        param_grid.update({'RegU': [1], 'RegI':[1]})
    elif DATASET == 'book-crossing':
        param_grid.update({'RegU': [1], 'RegI':[10]})
    
elif algorithm == 'MF':
    #MF
#    param_grid = {'NumFactors': [10,20,40], 'Regularization' :[0.01, 0.015, 0.05, 0.1] , 'LearnRate': [0.01, 0.03, 0.05], 'epsilon':[0.001], 'NumIter':[50], 'grid_search':[1] }
    param_grid = {'NumFactors': [10], 'Regularization' :[0.01, 0.015, 0.05, 0.1, 0.15, 0.5], 'LearnRate': [0.001, 0.005, 0.01, 0.02], 'epsilon':[0.01], 'NumIter':[100], 'grid_search':[1] }
    param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'Regularization' :[0.1], 'LearnRate': [0.01], 'epsilon':[0.001], 'NumIter':[100], 'grid_search':[0] }
    
    if DATASET == 'ml-1M':
        param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'Regularization': [0.05], 'epsilon': [0.001], 'grid_search': [1], 'LearnRate': [0.01], 'NumIter': [100]}
    
    if DATASET == 'jester-1':
#        param_grid = {'NumFactors': [10], 'Regularization' :[0.01, 0.015, 0.05, 0.1, 0.5], 'LearnRate': [0.0001, 0.0005, 0.001], 'epsilon':[0.001], 'NumIter':[100], 'grid_search':[1] }
        param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'Regularization' :[0.1], 'LearnRate': [0.0005], 'epsilon':[0.001], 'NumIter':[100], 'grid_search':[1] }
    
    if DATASET == 'book-crossing':
        param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'Regularization': [0.1], 'epsilon': [0.001], 'grid_search': [1], 'LearnRate': [0.03], 'NumIter': [100]}

elif algorithm == 'BMF':
    #BiasedMF
#    param_grid = {'NumFactors': [5, 10, 20, 40], 'BiasReg': [0.01, 0.1, 0.5], 'LearnRate': [0.005, 0.01, 0.05, 0.07], 'RegU': [0.01, 0.1, 0.5], 'RegI': [0.01, 0.1, 0.5],  'BoldDriver': [True, False], 'epsilon':[0.001], 'grid_search':[1] }
    #param_grid = {'NumFactors': [100, 120, 140], 'BiasReg': [0.1], 'LearnRate': [0.07], 'RegU': [1.0], 'RegI': [1.2], 'FrequencyRegularization' :[True],   'BoldDriver': [True], 'epsilon':[0.01], 'grid_search':[1] }
#    param_grid = {'NumFactors': [5, 10], 'BiasReg': [0.1, 0.2], 'NumIter': [100] , 'epsilon':[0.00001], 'grid_search':[1]}
#    param_grid = {'NumFactors': [5, 6], 'BiasReg': [0.25], 'FrequencyRegularization' :[True] , 'LearnRate': [0.03], 'RegU': [0.4], 'RegI': [1.2],  'BoldDriver': [False], 'epsilon':[0.001], 'NumIter':[80], 'grid_search':[0] }
#    param_grid = {'NumFactors': [5], 'BiasReg': [0.2], 'RegU': [0.2], 'RegI': [0.2], 'LearnRate': [0.005], 'NumIter': [100] , 'epsilon':[0.0001], 'grid_search':[0]}
    if DATASET == 'ml-100k':
         param_grid = {'NumFactors': [10], 'BiasReg': [0.01, 0.1, 0.2, 0.5], 'LearnRate': [0.001, 0.005, 0.01, 0.02], 'RegU': [0.01, 0.1, 0.5], 'RegI': [0.01, 0.1, 0.5], 'FrequencyRegularization' :[False, True],  'BoldDriver': [False], 'epsilon':[0.001], 'grid_search':[1] }
         param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'BiasReg': [0.2], 'FrequencyRegularization': [False], 'epsilon': [0.001], 'RegU': [0.1], 'grid_search': [1], 'BoldDriver': [False], 'LearnRate': [0.005], 'RegI': [0.1]}
    
    if DATASET == 'ml-1M':
        #param_grid = {'NumFactors': [5, 10, 20, 40], 'BiasReg': [0.01, 0.1, 0.5], 'LearnRate': [0.005, 0.01, 0.05, 0.07], 'RegU': [0.01, 0.1, 0.5], 'RegI': [0.01, 0.1, 0.5],  'BoldDriver': [True, False], 'epsilon':[0.001], 'grid_search':[1] }
#        param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'BiasReg': [0.75], 'FrequencyRegularization': [True], 'epsilon': [0.01], 'RegU': [0.5], 'grid_search': [1], 'BoldDriver': [False], 'LearnRate': [0.07], 'RegI': [0.5], 'NumIter': [100]}
        param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'BiasReg': [0.2], 'epsilon': [0.001], 'RegU': [0.2], 'grid_search': [1], 'BoldDriver': [False], 'LearnRate': [0.01], 'RegI': [0.01], 'NumIter': [100]}
    if DATASET == 'jester-1':
#        param_grid = {'NumFactors': [10], 'BiasReg': [0.01, 0.1, 0.2, 0.5], 'LearnRate': [0.0001, 0.0005, 0.001], 'RegU': [0.01, 0.1, 0.5], 'RegI': [0.01, 0.1, 0.5], 'FrequencyRegularization' :[False, True],  'BoldDriver': [False], 'epsilon':[0.001], 'grid_search':[1] }
        param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'BiasReg': [0.5], 'LearnRate': [0.0001], 'RegU': [0.5], 'RegI': [0.5], 'FrequencyRegularization' :[False],  'BoldDriver': [False], 'epsilon':[0.001], 'grid_search':[1] }
        
    if DATASET == 'book-crossing':
        param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'BiasReg': [0.1], 'epsilon': [0.001], 'RegU': [0.1], 'grid_search': [1], 'BoldDriver': [False], 'LearnRate': [0.01], 'RegI': [0.1], 'NumIter': [100]}

elif algorithm == 'SVDpp':
    #SVD++
    if DATASET =='ml-1M':
#        param_grid = {'NumFactors': [10,20,40], 'BiasReg': [0.33], 'Regularization': [0.01, 0.05], 'FrequencyRegularization': [True], 'LearnRate': [0.005, 0.01], 'BiasLearnRate': [0.7], 'NumIter':[80], 'epsilon':[0.001], 'grid_search':[1] }
        param_grid = {'NumFactors': [20], 'BiasReg': [0.5, 0.6], 'Regularization': [0.05], 'FrequencyRegularization': [False], 'LearnRate': [0.005], 'BiasLearnRate': [0.07, 0.7], 'NumIter':[120], 'epsilon':[0.001], 'grid_search':[0] }
        #param_grid = {'NumFactors': [5,10,20,50], 'BiasReg': [0.05, 0.005], 'Regularization': [0.1, 0.5, 1], 'FrequencyRegularization': [True, False], 'LearnRate': [0.01, 0.02], 'BiasLearnRate': [0.07, 0.007],  'NumIter':[100], 'epsilon':[0.001], 'grid_search':[1] }
        param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'BiasReg': [0.33], 'Regularization': [0.05], 'FrequencyRegularization': [False], 'LearnRate': [0.005], 'BiasLearnRate': [0.7], 'NumIter':[120], 'epsilon':[0.001], 'grid_search':[1] }
        
    if DATASET == 'jester-1':
         param_grid = {'NumFactors': [10], 'BiasReg': [0.33], 'Regularization': [0.01, 0.05, 0.1, 0.5], 'FrequencyRegularization': [True], 'LearnRate': [0.005, 0.01], 'BiasLearnRate': [0.7], 'NumIter':[80], 'epsilon':[0.001], 'grid_search':[1] }
         param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'BiasReg': [0.33], 'FrequencyRegularization': [True], 'Regularization': [0.5], 'grid_search': [1], 'epsilon': [0.001], 'BiasLearnRate': [0.7], 'LearnRate': [0.005], 'NumIter': [80]}
         
    if DATASET == 'ml-100k':
        param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'BiasReg': [0.005], 'FrequencyRegularization': [True], 'Regularization': [1], 'grid_search': [1], 'epsilon': [0.001], 'BiasLearnRate': [0.07], 'LearnRate': [0.01], 'NumIter': [100]}
        
    if DATASET == 'book-crossing':
        param_grid = {'NumFactors': [10], 'BiasReg': [0.33], 'Regularization': [0.01, 0.05, 0.1, 0.5], 'FrequencyRegularization': [True, False], 'LearnRate': [0.005, 0.01], 'BiasLearnRate': [0.7], 'NumIter':[100], 'epsilon':[0.001], 'grid_search':[1] }
#        param_grid = {'BiasReg': [0.005], 'FrequencyRegularization': [True], 'Regularization': [1], 'grid_search': [0], 'epsilon': [0.001], 'BiasLearnRate': [0.07], 'LearnRate': [0.01], 'NumFactors': [50], 'NumIter': [100]}
#        param_grid = {'NumFactors': [10], 'BiasReg': [0.05], 'Regularization': [0.05], 'FrequencyRegularization': [False], 'LearnRate': [0.005], 'BiasLearnRate': [0.005], 'NumIter':[10], 'epsilon':[0.001], 'grid_search':[0] }
        param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'BiasReg': [0.33], 'FrequencyRegularization': [False], 'Regularization': [0.5], 'grid_search': [1], 'epsilon': [0.001], 'BiasLearnRate': [0.7], 'LearnRate': [0.005], 'NumIter': [100]}

elif algorithm == 'FactorWiseMF':
    param_grid = {'NumFactors': [5, 10, 20, 40], 'Shrinkage':[20, 50, 100, 120, 150], 'NumIter':[10], 'grid_search':[1]}
    
    if DATASET == 'ml-100k':
        param_grid.update({'RegU': [5], 'RegI':[1]})
    elif DATASET =='ml-1M':
        param_grid.update({'RegU': [5], 'RegI':[1]})
    elif DATASET == 'jester-1':
        param_grid.update({'RegU': [1], 'RegI':[1]})
    elif DATASET == 'book-crossing':
        param_grid.update({'RegU': [1], 'RegI':[10]})
        
        
def selectAlgorithm(algo, params):
    if algo == 'MF':
        model = MF.MatrixFactorizer(**params)
    elif algo == 'BMF':
        model = MF.BiasedMatrixFactorizer(**params)
    elif algo == 'SVDpp':
        model = MF.SVDPlusPlusMF(**params)
    elif algo == 'FactorWiseMF':
        model = MF.FactorWiseMF(**params)
    elif algo =='GlobalAverage':
        model = Baselines.GlobalAverage()
    elif algo =='UserAverage':
        model = Baselines.UserAverage()
    elif algo =='ItemAverage':
        model = Baselines.ItemAverage()
    elif algo =='UserItemBaseline':
        model = MF.UserItemBaseline(**params)
    elif algo =='UserKNN':
        model = KNN.UserKNN(**params)
        
    return model


params = list(ParameterGrid(param_grid))


# grid search
grid_scores = list()
print "Fitting " + str(len(params)) + " parameter combinations\n"
par_idx=0
for par in params:
    par_idx+=1
    print "Fit " + str(par_idx) + "/" + str(len(params))
    
    
    model = selectAlgorithm(algorithm, par)
    model_name = algorithm
    
    ## DEBUG
#    model.rec.NumFactors
#    model.rec.BiasReg
#    model.rec.Regularization 
#    model.rec.FrequencyRegularization
#    model.rec.LearnRate
#    model.rec.BiasLearnRate
#    model.rec.NumIter
#    model.rec.Decay
#    model.epsilon
#    model.grid_search
    ## end DEBUG
    
    rmse, mae, time, rmse_log, mae_log, train_error_log = model.fit(DATASET, model_name, show_per_folds)


    grid_scores.append(CVScoreTuple(
                par,
                np.array(rmse, dtype=np.single).mean(),
                np.array(rmse, dtype=np.single).std(),
                np.array(mae, dtype=np.single).mean(),
                np.array(mae, dtype=np.single).std(),
                np.array(time, dtype=np.single).mean(),
                np.array(rmse),
                np.array(mae)))
    
    grid_scores_sorted = sorted(grid_scores, key=lambda x: x.mean_validation_score)
    
    # write file on every param combination in case of crash
    write2csv(log_dir + DATASET + '_' + model_name + '_gs_log.txt', grid_scores)
    write2csv(result_dir + DATASET + '_' + model_name + '_gs_sorted.txt', grid_scores_sorted)
    
    if rmse_log != 0:
        write2csv(log_dir + DATASET + '_' + model_name + '_rmse_log' + str(par_idx) + '.txt', rmse_log)
        write2csv(log_dir + DATASET + '_' + model_name + '_mae_log' + str(par_idx) + '.txt', mae_log)
        write2csv(log_dir + DATASET + '_' + model_name + '_train_log' + str(par_idx) + '.txt', train_error_log)

# write parameters grid to file
if len(param_grid) > 0:
    write_dict2csv(result_dir + DATASET + '_' + model_name + '_gs_params.txt', param_grid)

#a = read_dict_from_csv(result_dir + DATASET + '_' + model_name + '_gs_params.txt')
# get the model with the best parameters
best = grid_scores_sorted[0]    
print "\nBest scoring model"
print best

# -----end grid search---------------


# retrain model with best params
best_params = best.parameters

if len(best_params) > 0 and best_params['grid_search'] == 1:
    print "Refitting best model on full training data"
    best_params['grid_search'] = 0
    
    model = selectAlgorithm(algorithm, best_params)
    best_rmse, best_mae, best_time, b_rmse_log, b_mae_log, b_train_error_log = model.fit(DATASET, model_name, show_per_folds)
    
    best_score = []
    best_score.append(CVScoreTuple(
                    best_params,
                    np.array(best_rmse, dtype=np.single).mean(),
                    np.array(best_rmse, dtype=np.single).std(),
                    np.array(best_mae, dtype=np.single).mean(),
                    np.array(best_mae, dtype=np.single).std(),
                    np.array(best_time, dtype=np.single).mean(),
                    np.array(best_rmse),
                    np.array(best_mae)))
    
    print "\nBest scoring model on full training data"
    print best_score
    print algorithm, DATASET
    
    write2csv(result_dir + DATASET+ '_'+ model_name + '_best.txt', best_score)
    
    if b_rmse_log != 0:
        write2csv(result_dir + DATASET + '_' + model_name + '_best_rmse_log.txt', b_rmse_log)
        write2csv(result_dir + DATASET + '_' + model_name + '_best_mae_log.txt', b_mae_log)
        write2csv(result_dir + DATASET + '_' + model_name + '_best_train_log.txt', b_train_error_log)
    

    
    

