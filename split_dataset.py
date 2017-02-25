# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 01:01:27 2016

@author: George
"""

import pandas as pd
#from sklearn.cross_validation import KFold, ShuffleSplit
from sklearn.model_selection import KFold, ShuffleSplit

import rec_datasets
from tools.utils import check_mkdir

import os
user_path = os.environ['USERPROFILE']
data_path = user_path + r'\\Google Drive\\thesis\\datasets'

DATASET = 'ml-100k'
#DATASET = 'ml-1M'
#DATASET = 'jester-1'
#DATASET = 'book-crossing'

plot = False
plot = True

split_dataset = False

store_path = r'.\\data\\'+ DATASET + r'\\'
plots_path = r'.\\plots\\'

# check directories for existence (if not exist -> create)
check_mkdir(store_path)
check_mkdir(plots_path)
# ========================================================

 ## Load the dataset in pandas dataframe
data = rec_datasets.load_dataset(data_path, DATASET)

user_col, item_col = 'user_id', 'item_id'

def plot_distribution(user_col, item_col):
    # count the ratings of each user
    user_ratings_count = data[user_col].value_counts()
    # find the frequency of ratings of each user and sort by frequency
    user_ratings_count_freq = user_ratings_count.value_counts()
    user_ratings_count_freq.sort_index(inplace=True)
    
    item_ratings_count = data[item_col].value_counts()
    item_ratings_count_freq = item_ratings_count.value_counts()
    item_ratings_count_freq.sort_index(inplace=True)
    
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    plt.rc('font', family='Arial')
    fig, ax = plt.subplots()
    ax.loglog(user_ratings_count_freq, label=u'Χρήστες', linewidth=1)
    ax.loglog(item_ratings_count_freq, label=u'Αντικείμενα', linewidth=1)
    
    ax.set_xlabel(u'Πλήθος αξιολογήσεων')
    ax.set_ylabel(u'Πλήθος')
    ax.set_title(u'Κατανομή πλήθους χρηστών και αντικειμένων ανά πλήθος αξιολογήσεων', size=13)
    legend = ax.legend()
    for label in legend.get_texts():
        label.set_fontsize('large')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    plt.show()
    fig.savefig(plots_path + DATASET + '_dist')

if DATASET == 'ml-100k' or DATASET == 'ml-1M':
     data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
     if plot:
         plot_distribution('user_id', 'item_id')
     
if DATASET == 'jester-1':
     data.columns = ['user_id', 'item_id', 'rating']
     if plot:
         plot_distribution('user_id', 'item_id')
     
     
if DATASET == "book-crossing":
    data.rename(columns={'User-ID':'user_id', 'Book-Rating':'book_rating'}, inplace=True)
    temp_df = pd.DataFrame({'ISBN': data.ISBN.unique(), \
            'ISBN_id':range(len(data.ISBN.unique()))}) # map ids to unique ISBNs
    data = data.merge(temp_df, on='ISBN', how='left') # merge the mapped col to data
    cols = ['user_id', 'ISBN_id', 'book_rating', 'ISBN'] # rearrange cols
    data = data[cols]   # apply arrangement
    # remove implicit ratings (book_rating = 0)
    data = data.query('book_rating != 0')
    del data['ISBN']
    
    # remove users with less than 6 ratings
    counts = data['user_id'].value_counts()
    data = data[data['user_id'].isin(counts[counts >= 6].index)]
    if plot:
        plot_distribution('user_id', 'ISBN_id')


if split_dataset:
    num_folds = 5
    
    cv = KFold(len(data), n_folds=num_folds, shuffle=True, random_state=0)
    
    
    cv_index = 0
    for train_idx, test_idx in cv:
        cv_index = cv_index + 1
            
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
    
        filename = DATASET + '-f'
        
        train.to_csv(store_path + filename + str(cv_index) + '-train.csv', sep=' ', header=False, index=False)
        test.to_csv(store_path + filename + str(cv_index) + '-test.csv', sep=' ', header=False, index=False)
        
        val_split = ShuffleSplit(len(train), n_iter=1, test_size=0.125, random_state=0)
        
        for train_set_idx, val_set_idx in val_split:
            train_set = train.iloc[train_set_idx]
            val_set = train.iloc[val_set_idx]
            
            train_set.to_csv(store_path + filename + str(cv_index) + '-train_set.csv', sep=' ', header=False, index=False)
            val_set.to_csv(store_path + filename + str(cv_index) + '-val_set.csv', sep=' ', header=False, index=False)
