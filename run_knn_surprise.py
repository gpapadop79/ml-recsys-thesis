"""
This module describes how to manually train and test an algorithm without using
the evaluate() function.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import numpy as np

from surprise import KNNBasic
from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader
from surprise import evaluate

from tools.utils import CVScoreTuple
from tools.utils import write2csv
from tools.utils import check_mkdir
from tools import timing as tim

DATASET = 'ml-100k'
#DATASET = 'ml-1M'
#DATASET = 'jester-1'
#DATASET = 'book-crossing'

algorithm = 'UserKNN-Cosine'
#algorithm = 'ItemKNN-Cosine'
#algorithm = 'UserKNN-Pearson'
#algorithm = 'ItemKNN-Pearson'

baseline = True
#baseline = False


# Read command line arguments
if len(sys.argv) > 1:
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])
    
    DATASET = str(sys.argv[1])
    algorithm = str(sys.argv[2])
    if str(sys.argv[3]) == '1':
        baseline = True
    elif str(sys.argv[3]) == '0':
        baseline = False
    else:
        sys.exit('Invalid baseline arg')

folder = 'KNN_2'
result_dir = '.\\results\\'+ DATASET +'\\' + folder + '\\'
model_name = algorithm

# check directories for existence (if not exist -> create)
check_mkdir(result_dir)

# Prepare Data
files_dir = '.\\data\\' + DATASET + '\\'

reader = Reader(line_format='user item rating timestamp', sep=' ')
if DATASET == 'jester-1':
    reader = Reader(line_format='user item rating', sep=' ', rating_scale=(-10, 10))
if DATASET == 'book-crossing':
    reader = Reader(line_format='user item rating', sep=' ', rating_scale=(1, 10))

# folds_files is a list of tuples containing file paths:
train_file = files_dir + DATASET + '-f%d-train.csv'
test_file = files_dir + DATASET + '-f%d-test.csv'

folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]
data = Dataset.load_from_folds(folds_files, reader=reader)


if model_name == 'UserKNN-Cosine':
    sim_options = {'name': 'cosine',
                   'user_based': True
                   }
if model_name == 'ItemKNN-Cosine':
    sim_options = {'name': 'cosine',
                   'user_based': False  # compute  similarities between items
                   }
if model_name == 'UserKNN-Pearson':
    sim_options = {'name': 'pearson',
                   'user_based': True
                   }
if model_name == 'ItemKNN-Pearson':
    sim_options = {'name': 'pearson',
                   'user_based': False  # compute  similarities between items
                   }

bsl_options = {'method': 'als',
               'n_epochs': 10,
               }
               
if DATASET == 'ml-100k':
    bsl_options.update({'reg_u': 5, 'reg_i':1})
elif DATASET =='ml-1M':
    bsl_options.update({'reg_u': 5, 'reg_i':1})
elif DATASET == 'jester-1':
    bsl_options.update({'reg_u': 1, 'reg_i':1})
elif DATASET == 'book-crossing':
    bsl_options.update({'reg_u': 1, 'reg_i':10})
               

k_params = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

#k_params = [1, 3, 5]

print()          
print(DATASET)

if baseline:
    model_name = 'basel-' + model_name

scores = list()
for k in k_params:
    
    if baseline:
        algo = KNNBaseline(k=k, sim_options=sim_options, bsl_options=bsl_options)
    else:   
        algo = KNNBasic(k=k, sim_options=sim_options)
    
    print('K =',k)
    tim.startlog('Training model')
    # Evaluate performances of our algorithm on the dataset.
    result = evaluate(algo, data)
    rmse = result['rmse']
    mae = result['mae']
    time = tim.endlog('Done training model')
    
    params = {'k':k}
    params.update({'sim_options': sim_options})
    if baseline:
        params.update({'bsl_options': bsl_options})
        
    scores.append(CVScoreTuple(
                params,
                np.array(rmse, dtype=np.single).mean(),
                np.array(rmse, dtype=np.single).std(),
                np.array(mae, dtype=np.single).mean(),
                np.array(mae, dtype=np.single).std(),
                np.array(time / 5, dtype=np.single).mean(),
                np.array(rmse),
                np.array(mae)))
    
    write2csv(result_dir + DATASET + '_' + model_name + '_score_log.txt', scores)
    scores_sorted = sorted(scores, key=lambda x: x.mean_validation_score)
    write2csv(result_dir + DATASET + '_' + model_name + '_score_sorted_log.txt', scores_sorted)
    
print('Finished', model_name, 'on', DATASET)
