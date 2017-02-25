# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 18:34:38 2017

@author: George
"""

from tools.utils import read_csv
import numpy as np
import pandas as pd
import ast


def read_data(DATASET, filename, folder):

#    folder = 'KNN'
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
       k.append(params['NumFactors'])
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
    
def make_plot(DATASET, folder, files, baselines, baseline_mask=[], rmse_limits = []):

    markers = ['s', 'o', 'D', '^']
    colors = ['b', 'm', 'r', 'grey']
    
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    plt.rc('font', family='Arial')
    fig, ax = plt.subplots()
    ax.set_xlabel(u'k: Αριθμός παραγόντων')
    ax.set_ylabel(u'RMSE')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.xlim([0, 125])
    if len(rmse_limits) > 0:
        plt.ylim(rmse_limits)
    
    i=0
    for label in np.sort(files.keys()):
        df = read_data(DATASET, files[label], folder)
        k = np.array(df['k'], dtype=np.int)
        rmse = np.array(df['rmse'], dtype=np.float)
    
        ax.plot(k, rmse, label=label, linewidth=1.5, marker=markers[i], markersize=5)#, color=colors[i])
        
        #find minimum of curve
        min_y = df['rmse'].min()
        min_k = df[df['rmse']==min_y]['k']
        ax.plot(min_k, min_y, 'o', markersize=12, markeredgewidth=1, markerfacecolor='none', color='black') # circle the minimum point
        i+=1
    
    #plot baselines
    if len(baseline_mask) == 0:
        baseline_mask = {'u_avg': 0, 'i_avg': 0, 'ui_baseline': 0} # default values if not mask is given
    
    if baseline_mask['ui_baseline'] == 1:
        ui_baseline = np.ones(len(k)) * baselines['ui_baseline']
        ax.plot(k, ui_baseline, label='U-I Baseline', linewidth=1.5, ls='dashed', marker='+', markersize=5, color='g')
    
    if baseline_mask['u_avg'] == 1:
        u_avg = np.ones(len(k)) * baselines['u_avg']
        ax.plot(k, u_avg, label='User Avg', linewidth=1.5, ls='dashed', marker='+', markersize=5, color='brown')
    
    if baseline_mask['i_avg'] == 1:
        i_avg = np.ones(len(k)) * baselines['i_avg']
        ax.plot(k, i_avg, label='Item Avg', linewidth=1.5, ls='dashed', marker='+', markersize=5, color='y')
    
    # ticks every 10 values of k
#    plt.xticks(np.arange(min(k+5), max(k)+1, 10.0))
    plt.xticks([5, 10, 20, 30, 40, 50, 60, 80, 100, 120])
    
   
#    legend = ax.legend().get_frame().set_alpha(0.2)
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
DATASET = 'jester-1'
#DATASET = 'book-crossing'

if DATASET == 'ml-100k':
    folder = ''
#    baselines = {'u_avg':1.0419, 'i_avg':1.0249, 'ui_baseline': 0.9403}
    baseline_mask = {'u_avg': 0, 'i_avg': 0, 'ui_baseline': 0}
    RMSE_limits = [0.91, 0.95]
    RMSE_limits = []
                 
    files1 = {'MF': '\\final\\ml-100k_MF-run2\\ml-100k_MF_gs_sorted.txt',
              'BMF':'\\final\\BMF\\run2\\ml-100k_BMF_gs_sorted.txt',
              'SVD++': '\\final\\SVDpp\\run1\\ml-100k_SVDpp_gs_sorted.txt',
              'ALS': '\\final\\ALS\\run2\\ml-100k_ALS_score_sorted_log.txt'}
             
            
if DATASET == 'ml-1M':
    folder = ''
    baselines = {'u_avg':1.0355, 'i_avg':0.9795, 'ui_baseline': 0.9077}
    baseline_mask = {'u_avg': 0, 'i_avg': 0, 'ui_baseline': 0}
    RMSE_limits = [0.855, 0.875]
    RMSE_limits = [0.855, 0.96]
#    RMSE_limits = []
    
    files1 = {'MF': '\\final\\MF\\run1\\ml-1M_MF_gs_sorted.txt',
              'BMF':'\\final\\BMF\\run3\\ml-1M_BMF_gs_sorted.txt',
              'SVD++': '\\final\\SVDpp\\ml-1M_SVDpp_gs_sorted.txt',
              'ALS': '\\final\\ALS\\run2\\ml-1M_ALS_score_sorted_log.txt'}

if DATASET == 'jester-1':
    folder = ''
    baselines = {'u_avg':4.6323, 'i_avg':4.9870, 'ui_baseline': 4.3390}
    baseline_mask =  {'u_avg': 0, 'i_avg': 0, 'ui_baseline': 1}
    RMSE_limits = [4, 5.05]
#    RMSE_limits = []
    
    files1 = {'MF': '\\final\\jester-1_MF_run2 (various f)\\jester-1_MF_gs_sorted.txt',
              'BMF':'\\final\\jester-1_BMF_run3 (various f)\\jester-1_BMF_gs_sorted.txt',
              'SVD++': '\\final\\SVDpp\\run2\\jester-1_SVDpp_gs_sorted.txt',
              'ALS': '\\final\\ALS\\run3\\jester-1_ALS_score_sorted_log.txt'}
              
if DATASET == 'book-crossing':
    baselines = {'u_avg':1.6055, 'i_avg':1.9547, 'ui_baseline': 1.5724}
    baseline_mask = {'u_avg': 0, 'i_avg': 0, 'ui_baseline': 1}
    
    RMSE_limits = [1.55, 1.91]
#    RMSE_limits = []
    
    files1 = {'MF': '\\MF\\book-crossing_MF_gs_sorted.txt',
              'BMF':'\\final\\BMF\\run1\\book-crossing_BMF_gs_sorted.txt',
              'SVD++': '\\final\\SVDpp\\run2\\book-crossing_SVDpp_gs_sorted.txt',
              'ALS': '\\final\\ALS\\run2\\book-crossing_ALS_score_sorted_log.txt'}
 

         
make_plot(DATASET, folder, files1, baselines, baseline_mask, RMSE_limits)


