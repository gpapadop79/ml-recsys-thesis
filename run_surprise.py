"""
This module descibes how to load a custom dataset when folds (for
cross-validation) are predefined by train and test files.

As a custom dataset we will actually use the movielens-100k dataset, but act as
if it were not built-in.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise import evaluate
from surprise import Reader

from tools import timing as tim

# path to dataset folder
files_dir = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/')
files_dir = '.\\data\\ml-100k\\'
#files_dir = '.\\data\\ml-1M\\'
#files_dir = '.\\data\\jester-1\\'
#files_dir = '.\\data\\book-crossing\\'

# This time, we'll use the built-in reader.
reader = Reader(line_format='user item rating timestamp', sep=' ')
#reader = Reader(line_format='user item rating', sep=' ', rating_scale=(-10, 10))
#reader = Reader(line_format='user item rating', sep=' ', rating_scale=(1, 10))


# folds_files is a list of tuples containing file paths:
# [(u1.base, u1.test), (u2.base, u2.test), ... (u5.base, u5.test)]
#train_file = files_dir + 'u%d.base'
#test_file = files_dir + 'u%d.test'

train_file = files_dir + 'ml-100k-f%d-train.csv'
test_file = files_dir + 'ml-100k-f%d-test.csv'

#train_file = files_dir + 'ml-1M-f%d-train.csv'
#test_file = files_dir + 'ml-1M-f%d-test.csv'

#train_file = files_dir + 'jester-1-f%d-train.csv'
#test_file = files_dir + 'jester-1-f%d-test.csv'
##
#train_file = files_dir + 'book-crossing-f%d-train.csv'
#test_file = files_dir + 'book-crossing-f%d-test.csv'

folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

data = Dataset.load_from_folds(folds_files, reader=reader)

# We'll use an algorithm that predicts baseline estimates.
bsl_options = {'method': 'sgd',
               'learning_rate': .005,
               }
bsl_options = {'method': 'als',
               'reg_u': 1,
               'reg_i': 10}
algo = BaselineOnly(bsl_options=bsl_options)

sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBasic(k=10, sim_options=sim_options)

svd_options = {'n_factors':10, 'biased': True, 'lr_all': 0.01, 'reg_all': 0.2  }
#algo = SVD(**svd_options)

#algo = SVDpp(n_factors=10, lr_all=0.005, reg_all=0.05, n_epochs=10)

tim.startlog('Training model')
# Evaluate performances of our algorithm on the dataset.
evaluate(algo, data)
tim.endlog('Done training model')
