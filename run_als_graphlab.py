### Load Data ###
# MovieLens dataset collected by the GroupLens Research Project at the University of Minnesota.
# For more information, see http://grouplens.org/datasets/movielens/

from os import path
import graphlab as gl
import numpy as np
from datetime import datetime
import sys

from tools.utils import CVScoreTuple
from tools.utils import write2csv
from tools.utils import check_mkdir
from sklearn.grid_search import ParameterGrid


DATASET = 'ml-100k'
#DATASET = 'ml-1M'
#DATASET = 'jester-1'
#DATASET = 'book-crossing'

# dataset scale bounds
c_low = 1
c_high = 5
if DATASET == 'jester-1':
    c_low = -10
    c_high = 10
if DATASET == 'book-crossing':
    c_low = 1
    c_high = 10

algorithm = 'ALS'
model_name = algorithm

if len(sys.argv)>1:
    print sys.argv[1]
    
    DATASET = str(sys.argv[1])

files_dir = '.\\data\\' + DATASET + '\\'
result_dir = '.\\results\\'+ DATASET +'\\' + algorithm + '\\'

# check directories for existence (if not exist -> create)
check_mkdir(result_dir)

### Train Recommender Model ###
train_file = files_dir + DATASET + '-f%d-train.csv'
test_file = files_dir + DATASET + '-f%d-test.csv'

folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]


if DATASET=='jester-1':
#    param_grid = {'NumFactors': [10], 'reg':[1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
#    param_grid = {'NumFactors': [10], 'reg':[ 1.2e-4, 1.4e-4, 1.6e-4, 1.8e-4]}
    param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'reg':[2e-4]}

if DATASET=='book-crossing':
#    param_grid = {'NumFactors': [10], 'reg':[1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
    param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'reg':[5.8e-5]}
#    param_grid = {'NumFactors': [10], 'reg':[2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5]}
#    param_grid = {'NumFactors': [10], 'reg':[ 5.5e-5, 5.8e-5, 6e-5, 6.2e-5, 6.5e-5]}
    param_grid = {'NumFactors': [10], 'reg':[5.8e-5], 'lin_reg':[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}

if DATASET=='ml-1M':
#    param_grid = {'NumFactors': [10], 'reg':[0.6e-5, 0.8e-5, 0.9e-5, 1e-5, 1.1e-5, 1.2e-5, 1.4e-5]}
    param_grid = {'NumFactors': [20], 'reg':[1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
#    param_grid = {'NumFactors': [5, 10, 20, 30, 40, 50, 60, 80, 100, 120], 'reg':[8e-6]}

if DATASET=='ml-100k':
    param_grid = {'NumFactors': [10], 'reg':[1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
    aram_grid = {'NumFactors': [10], 'reg':[1e-4]}



params = list(ParameterGrid(param_grid))


scores=list()

for par in params:

    fold_rmse = []
    fold_mae = []
    fold_time = []
    for (train,test) in folds_files:
        
        train_data = gl.SFrame.read_csv(train, delimiter =' ', header = False)
        if DATASET=='jester-1' or DATASET=='book-crossing':
            train_data.rename({'X1' : 'userId', 'X2' : 'movieId', 'X3' : 'rating'})
        else:
            train_data.rename({'X1' : 'userId', 'X2' : 'movieId', 'X3' : 'rating', 'X4' : 'timestamp'})
            
            
        val_data = gl.SFrame.read_csv(test, delimiter =' ', header = False)
        if DATASET=='jester-1' or DATASET=='book-crossing':
            val_data.rename({'X1' : 'userId', 'X2' : 'movieId', 'X3' : 'rating'})
        else:
            val_data.rename({'X1' : 'userId', 'X2' : 'movieId', 'X3' : 'rating', 'X4' : 'timestamp'})
        
        
        model = gl.factorization_recommender.create(train_data, 'userId', 'movieId', target='rating', nmf = False,
                                                    num_factors=par['NumFactors'], solver='als', max_iterations=15, 
                                                    regularization=par['reg'], 
                                                    side_data_factorization=False, random_seed = 0 )
        
        
#        rmse = model.evaluate_rmse(val_data, target='rating')
#        print 'RMSE =', rmse['rmse_overall']
#        fold_rmse.append(rmse['rmse_overall'])
        
        predictions = model.predict(val_data)
        predictions = predictions.clip(lower=c_low, upper=c_high)
        
        residuals = val_data['rating'] - predictions
        
        rmse = ((residuals**2).mean())**0.5
        print 'RMSE =', rmse
        fold_rmse.append(rmse)
        
        mae = np.abs(residuals).mean()
        print 'MAE =', mae
        fold_mae.append(mae)
        
        print 'training time =',model.training_time
        fold_time.append(model.training_time)
        
    scores.append(CVScoreTuple(
                par,
                np.array(fold_rmse, dtype=np.single).mean(),
                np.array(fold_rmse, dtype=np.single).std(),
                np.array(fold_mae, dtype=np.single).mean(),
                np.array(fold_mae, dtype=np.single).std(),
                np.array(fold_time, dtype=np.single).mean(),
                np.array(fold_rmse),
                np.array(fold_mae)))
                
    write2csv(result_dir + DATASET + '_' + model_name + '_score_log.txt', scores)
    scores_sorted = sorted(scores, key=lambda x: x.mean_validation_score)
    write2csv(result_dir + DATASET + '_' + model_name + '_score_sorted_log.txt', scores_sorted)
        
