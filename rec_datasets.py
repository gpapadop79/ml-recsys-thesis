# -*- coding: utf-8 -*-
"""
Created on Tue May 10 23:46:00 2016

@author: George

Reads a dataset by it's name
"""

import pandas as pd
import numpy as np

def load_dataset(data_path, dataset):
    """ Reads a dataset """

    if dataset == 'ml-100k':
        folder = r'ml-100k\u.data'
        separator = '\t'
        header = None

    elif dataset == 'ml-1M':
        # movielens 1M
        folder = r'ml-1m\ratings1.dat'
        separator = '::'
        header = None

    elif dataset == 'ml-10M':
        # movielens 10M
        folder = r'ml-10m\ratings.dat'
        separator = '::'
        header = None

    elif  dataset == 'jester-1':
        # jester-1
        folder = r'jester\jester-1-ratings.txt'
        separator = '\t'
        header = None

    elif dataset == 'jester-4M':
        # jester-4M
        folder = r'jester\jester-full-ratings.txt'
        separator = '\t'
        header = None

    elif  dataset == 'book-crossing':
        # book crossing
        folder = r'book-crossing\BX-Book-Ratings.csv'
        separator = ';'
        header = 0
    
    elif  dataset == 'epinions':
        # epinions
        folder = r'epinions-rating\out.epinions-rating'
        separator = ' '
        header = 0
    
    elif  dataset == 'amazon-ratings':
        # amazon ratings
        folder = r'amazon-ratings\out.amazon-ratings'
        separator = ' '
        header = 0
        
    elif  dataset == 'eachmovie':
        # eachmovie
        folder = r'eachmoviedata\vote.txt'
        separator = '\t'
        header = None
    
    elif  dataset == 'rec-eachmovie':
        # rec-eachmovie
        folder = r'rec-eachmovie\rec-eachmovie.edges'
        separator = ' '
        header = None
    
    elif  dataset == 'netflix':
#        folder = r'netflix\netflix_mme.txt'
        folder = r'C:\Users\Vasso\Desktop\code.graphlab.datasets\netflix_mm.txt'
#        folder = r'D:\DATA\netflix_dataset\netflix_mm.txt'
        separator = ' '
        header = None
   

    elif dataset == 'hetrec2011-lastfm-2k':
         # hetrec2011 last.fm 2k
        folder = r'hetrec2011-lastfm-2k\user_artists.dat'
        separator = '\t'
        header = 0

    print 'Reading dataset ' + dataset
    
    if dataset == 'netflix':
#        data = pd.read_table(folder, sep=separator, header=header, skiprows=3)#, usecols=[0,1,3])
        data = pd.read_table(folder, sep=separator, header=header, usecols=[0,1,3], skiprows=3, #nrows=1000000,
                             names=['user', 'item', 'rating'], 
                             dtype={'user': np.int32, 'item': np.int32, 'rating': np.int32})
    else:
        data = pd.read_table(data_path + '\\' + folder, sep=separator, header=header)
        
#        data = pd.read_table(data_path + '\\' + folder, sep=separator, header=header, 
#                             names=['user', 'item', 'rating', 'timest'], 
#                             dtype={'user': np.int16, 'item': np.int32, 'rating': np.int32})
#        
    return data

    
