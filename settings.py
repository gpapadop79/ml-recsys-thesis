# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 20:57:00 2016

@author: George

Settings file
"""

import os

# === dataset to load =========================================================
#data_path = r'..\\datasets'
user_path = os.environ['USERPROFILE']
data_path = user_path + r'\\Google Drive\\thesis\\datasets'

    ## dataset name (each dataset path info in rec_datasets.py module)

    ## Short info ##

    # ml-100k: MovieLens 100k -> 100k ratings (1-5) of 943 users on 1682 movies
    # ml-1M: MovieLens 1M -> 1M ratings (1-5) of 6040 users on 3706 movies
    # ml-10M: MovieLens 10M -> 10M ratings (1-5) of 69878 users on 10677 movies
    # jester-1: Jester subset -> 1.8M continuous ratings (-10 to 10) of 24983 users on 100 jokes (min ratings per user 36)
    # jester-4M: Jester full -> 4.1M continuous ratings (-10 to 10) of 73421 users on 100 jokes (min ratings per user 15)
    # book-crossing:  Book Crossing -> 433k explicit ratings (1-10) of 77805 users on 185973 books
    # epinions:  Epinions -> 13.6M ratings (1-5) of 120492 users on 755760 products
    # amazon-ratings:  Amazon ratings -> 5.8M ratings (1-5) of 2.146.057 users on 1.230.915 products
    # eachmovie:  Eachmovie (original) -> 2.8M ratings (0 to 1 per 0.2) of 74424 users on 1648 movies
    # rec-eachmovie:  Eachmovie  -> 2.8M ratings (1-6) of 61265 users on 1623 movies

dataset = 'ml-100k'
#dataset = 'ml-1M'
#dataset = 'ml-10M'
#dataset = 'jester-1'
#dataset = 'jester-4M'
#dataset = 'book-crossing'
#dataset = 'epinions'
#dataset = 'amazon-ratings'
#dataset = 'eachmovie'
#dataset = 'rec-eachmovie'
#dataset = 'netflix'

# =============================================================================

# =============================================================================
## Train - Test split options ##

## set 1 for grid search or fast experiment (random hold-out split)
## set 5 or 10 for k-fold cross validation

num_folds = 1
#num_folds = 5

testset_size = 0.2 # only for num_folds = 1

# =============================================================================

# === truncatedSVD parameters =================================================
no_tsvd = 0 # set to 1 for no truncatedSVD
tsvd_components_users = 2
tsvd_components_items = 2
# =============================================================================


