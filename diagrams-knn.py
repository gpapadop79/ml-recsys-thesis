# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 18:34:38 2017

@author: George
"""

from tools.utils import read_csv
import numpy as np
import pandas as pd
import ast


def read_data(DATASET, filename):

    folder = 'KNN'
    result_dir = '.\\results\\'+ DATASET +'\\' + folder + '\\'

    data = read_csv(result_dir + filename)
    
    df = pd.DataFrame()
    
    k = list()
    rmse = list()
    rmse_std = list()
    mae = list()
    mae_std = list()
    for i in range(1, len(data)):
       params = ast.literal_eval(data[i][0]) 
       k.append(params['k'])
       rmse.append(data[i][1])
       rmse_std.append(data[i][2])
       mae.append(data[i][3])
       mae_std.append(data[i][4])
   
    #k = np.asarray(k, dtype=np.int)
    #rmse = np.asarray(rmse, dtype=np.float)
    #rmse_std = np.asarray(rmse_std, dtype=np.float)
    #mae = np.asarray(mae, dtype=np.float)
    #mae_std = np.asarray(mae_std, dtype=np.float)

    k = pd.Series(k)
    rmse = pd.Series(rmse)
    rmse_std = pd.Series(rmse_std)
    mae = pd.Series(mae)
    mae_std = pd.Series(mae_std)
    df['k'] = k.values
    df['rmse'] = rmse.values
    df['rmse_std'] = rmse_std.values
    df['mae'] = mae.values
    df['mae_std'] = mae_std.values

    df.sort_values(by='k', inplace=True) #sort the values for x-axis

    return df
    
def make_plot(DATASET, files, baselines, baseline_mask=[], rmse_limits = []):

    markers = ['s', 'o', 'D', '^']
    colors = ['b', 'm', 'r', 'grey']
    
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    plt.rc('font', family='Arial')
    fig, ax = plt.subplots()
    ax.set_xlabel(u'k: Πλήθος γειτόνων')
    ax.set_ylabel(u'RMSE')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.xlim([-3, 103])
    if len(rmse_limits) > 0:
        plt.ylim(rmse_limits)
    
    i=0
    for label in np.sort(files.keys()):
        df = read_data(DATASET, files[label])
        k = np.array(df['k'], dtype=np.int)
        rmse = np.array(df['rmse'], dtype=np.float)
    
        ax.plot(k, rmse, label=label, linewidth=1.5, marker=markers[i], markersize=5)#, color=colors[i])
        
        #find minimum of curve
        min_y = df['rmse'].min()
        min_k = df[df['rmse']==min_y]['k']
        if len(min_k) > 0:
            ax.plot(min_k.iloc[0], min_y, 'o', markersize=12, markeredgewidth=1, markerfacecolor='none', color='black') # circle the minimum point
        else:
            ax.plot(min_k, min_y, 'o', markersize=12, markeredgewidth=1, markerfacecolor='none', color='black') # circle the minimum point
        i+=1
    
    
    #plot baselines
    if len(baseline_mask) == 0:
        baseline_mask = {'u_avg': 1, 'i_avg': 1, 'ui_baseline': 1} # default values if not mask is given
    
    if baseline_mask['ui_baseline'] == 1:
        ui_baseline = np.ones(len(k)) * baselines['ui_baseline']
        ax.plot(k, ui_baseline, label='U-I Baseline', linewidth=1.5, ls='dashed', marker='+', markersize=5, color='g')
    
    if baseline_mask['u_avg'] == 1:
        u_avg = np.ones(len(k)) * baselines['u_avg']
        ax.plot(k, u_avg, label='User Avg', linewidth=1.5, ls='dashed', marker='+', markersize=5, color='brown')
    
    if baseline_mask['i_avg'] == 1:
        i_avg = np.ones(len(k)) * baselines['i_avg']
        ax.plot(k, i_avg, label='Item Avg', linewidth=1.5, ls='dashed', marker='+', markersize=5, color='y')
    
#    plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xticks([1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
    # plot legend
    legend = ax.legend()
    for label in legend.get_texts():
        label.set_fontsize('large')
        
    
    # sort both labels and handles by labels of the legend
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)
    plt.tight_layout()
    

#DATASET = 'ml-100k'
#DATASET = 'ml-1M'
#DATASET = 'jester-1'
DATASET = 'book-crossing'

if DATASET == 'ml-100k':
    baselines = {'u_avg':1.0419, 'i_avg':1.0249, 'ui_baseline': 0.9403}
    
    RMSE_limits = [0.90, 1.12]
    
    files1 = {'Biased UserKNN-Cosine':'ml-100k_basel-UserKNN-Cosine_score_log.txt',
         'Biased UserKNN-Pearson':'ml-100k_basel-UserKNN-Pearson_score_log.txt',
         'Biased ItemKNN-Cosine':'ml-100k_basel-ItemKNN-Cosine_score_log.txt',
         'Biased ItemKNN-Pearson':'ml-100k_basel-ItemKNN-Pearson_score_log.txt'}
         
    files2 = {'UserKNN-Cosine':'ml-100k_UserKNN-Cosine_score_log.txt',
             'UserKNN-Pearson':'ml-100k_UserKNN-Pearson_score_log.txt',
             'ItemKNN-Cosine':'ml-100k_ItemKNN-Cosine_score_log.txt',
             'ItemKNN-Pearson':'ml-100k_ItemKNN-Pearson_score_log.txt'}
             
    files3 = {'UserKNN-Cosine':'ml-100k_UserKNN-Cosine_score_log.txt',
             'UserKNN-Pearson':'ml-100k_UserKNN-Pearson_score_log.txt',
             'Biased UserKNN-Cosine':'ml-100k_basel-UserKNN-Cosine_score_log.txt',
             'Biased UserKNN-Pearson':'ml-100k_basel-UserKNN-Pearson_score_log.txt'}
             
             
    files4 = {'ItemKNN-Cosine':'ml-100k_ItemKNN-Cosine_score_log.txt',
             'ItemKNN-Pearson':'ml-100k_ItemKNN-Pearson_score_log.txt',
             'Biased ItemKNN-Cosine':'ml-100k_basel-ItemKNN-Cosine_score_log.txt',
             'Biased ItemKNN-Pearson':'ml-100k_basel-ItemKNN-Pearson_score_log.txt'}
    
if DATASET == 'ml-1M':
    baselines = {'u_avg':1.0355, 'i_avg':0.9795, 'ui_baseline': 0.9077}
    
    RMSE_limits = [0.85, 1.10]
    
    files3 = {'UserKNN-Cosine':'ml-1M_UserKNN-Cosine_score_log.txt',
             'UserKNN-Pearson':'ml-1M_UserKNN-Pearson_score_log.txt',
             'Biased UserKNN-Cosine':'ml-1M_basel-UserKNN-Cosine_score_log.txt',
             'Biased UserKNN-Pearson':'ml-1M_basel-UserKNN-Pearson_score_log.txt'}
             
             
    files4 = {'ItemKNN-Cosine':'ml-1M_ItemKNN-Cosine_score_log.txt',
             'ItemKNN-Pearson':'ml-1M_ItemKNN-Pearson_score_log.txt',
             'Biased ItemKNN-Cosine':'ml-1M_basel-ItemKNN-Cosine_score_log.txt',
             'Biased ItemKNN-Pearson':'ml-1M_basel-ItemKNN-Pearson_score_log.txt'}

if DATASET == 'jester-1':
    baselines = {'u_avg':4.6323, 'i_avg':4.9870, 'ui_baseline': 4.3390}
    
    RMSE_limits = []
                 
             
    files4 = {'ItemKNN-Cosine':'jester-1_ItemKNN-Cosine_score_log.txt',
             'Biased ItemKNN-Cosine':'jester-1_basel-ItemKNN-Cosine_score_log.txt',
             }

if DATASET == 'book-crossing':
    baselines = {'u_avg':1.6055, 'i_avg': 1.9547, 'ui_baseline': 1.5724}
    
    RMSE_limits = [1.55, 2.2]
    
    files4 = {
              'UserKNN-Cosine':'book-crossing_UserKNN-Cosine_score_log.txt',
             'UserKNN-Pearson':'book-crossing_UserKNN-Pearson_score_log.txt',
             'Biased UserKNN-Cosine':'book-crossing_basel-UserKNN-Cosine_score_log.txt',
             'Biased UserKNN-Pearson':'book-crossing_basel-UserKNN-Pearson_score_log.txt'
             }
             
    files5 = {
              'UserKNN-Cosine':'\\run2\\book-crossing_UserKNN-Cosine_score_log.txt',
             'UserKNN-Pearson':'\\run2\\book-crossing_UserKNN-Pearson_score_log.txt'
             }
    baseline_mask5 = {'u_avg': 0, 'i_avg': 1, 'ui_baseline': 0}
    RMSE_limits5 = []
    
    files6 = {
             'Biased UserKNN-Cosine':'\\run2\\book-crossing_basel-UserKNN-Cosine_score_log.txt',
             'Biased UserKNN-Pearson':'\\run2\\book-crossing_basel-UserKNN-Pearson_score_log.txt'
             }  
    baseline_mask6 = {'u_avg': 1, 'i_avg': 0, 'ui_baseline': 1}
    RMSE_limits6 = []
    
    
    files7 = {
              'UserKNN-Cosine':'\\run2\\book-crossing_UserKNN-Cosine_score_log.txt',
             'UserKNN-Pearson':'\\run2\\book-crossing_UserKNN-Pearson_score_log.txt',
             'Biased UserKNN-Cosine':'\\run2\\book-crossing_basel-UserKNN-Cosine_score_log.txt',
             'Biased UserKNN-Pearson':'\\run2\\book-crossing_basel-UserKNN-Pearson_score_log.txt'
             }
    baseline_mask7 = {'u_avg': 1, 'i_avg': 1, 'ui_baseline': 1}
    RMSE_limits7 = [1.55, 2.19]
             
             
#make_plot(DATASET, files1)
#make_plot(DATASET, files2)
#make_plot(DATASET, files3, baselines, RMSE_limits)
#make_plot(DATASET, files4, baselines, RMSE_limits)

#make_plot(DATASET, files5, baselines, baseline_mask5,  rmse_limits=RMSE_limits5)
#make_plot(DATASET, files6, baselines, baseline_mask6,  rmse_limits=RMSE_limits6)

make_plot(DATASET, files7, baselines, baseline_mask7,  rmse_limits=RMSE_limits7)


