"""
This module describes how to manually train and test an algorithm without using
the evaluate() function.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
from surprise import GridSearch
from surprise import SVD
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader

#DATASET = 'ml-100k'
DATASET = 'ml-1M'
#DATASET = 'jester-1'
#DATASET = 'book-crossing'

# Read command line arguments
if len(sys.argv) > 1:
    print(sys.argv[1])
    
    DATASET = str(sys.argv[1])

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

sim_options1 = {'name': 'cosine',
               'user_based': True  # compute  similarities between items
               }
sim_options2 = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
               
param_grid = {'k': [10, 20, 30, 40, 50, 60], 'sim_options':[sim_options1, sim_options2]}

print()          
print(DATASET)
grid_search = GridSearch(KNNBasic, param_grid, measures=['RMSE'])
grid_search.evaluate(data)

# best RMSE score
print(grid_search.best_score['RMSE'])

# combination of parameters that gave the best RMSE score
print(grid_search.best_params['RMSE'])

import pandas as pd

results_df = pd.DataFrame.from_dict(grid_search.cv_results)
print(results_df)
