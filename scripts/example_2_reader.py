#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:09:26 2025

@author: xbb
"""

import warnings, os, pickle
import numpy as np

#%%
n_features = 20 # 30
n_train = 500 # 500

hyper_ = [2, 3, 4, 5, 6]

# Get the file path
path_temp = os.getcwd()
result = path_temp.split("/")
path = ''
checker = True
for elem in result:
    if elem != 'oblique_trees' and checker:
        path = path + elem + '/'
    else:
        checker = False
path = path + 'oblique_trees' + '/'

all_reuslts = []
file = path + 'simulated_data/runtime' + str(n_features) + '_'  + str(n_train)

results_loaded = []
with open(file, 'rb') as f:
    results_loaded = pickle.load(f)

# A R X 4 matrix
# b in [2, 20, 200, 2000, 20000]
# s0 in [2, 3, 4, 5, 6]
all_reuslts = np.zeros([100, 5, 5])
for r_, current_results in enumerate(results_loaded):
    all_reuslts[r_] = current_results


results_loaded[0]
len(results_loaded)

# report data
A = np.round(np.mean(all_reuslts, axis = 0), 2)
B = np.round(np.std(all_reuslts, axis = 0), 2)


for s0 in range(5):
    print(" & ".join(f"{A[s0, i]} ({B[s0, i]})" for i in range(len(A[s0, :]))))

    
    
#%%
# RF-RC



file = path + 'simulated_data/runtime' + str(n_features) + '_' + str(n_train) + 'rfrc'

results_loaded = []
with open(file, 'rb') as f:
    results_loaded = pickle.load(f)

len(results_loaded)
# A R X 4 matrix

# s0 in [2, 3, 4, 5, 6]
# max_features in [100, 1000, 10000, 100000]
all_reuslts = np.zeros([100, 5, 4])
for r_, current_results in enumerate(results_loaded):
    all_reuslts[r_] = current_results




    

# report data
A = np.round(np.mean(all_reuslts, axis = 0), 2)
B = np.round(np.std(all_reuslts, axis = 0), 2)

for s0 in range(5):
    print(" & ".join(f"{A[s0, i]} ({B[s0, i]})" for i in range(len(A[s0, :]))))

#     #%% 
#     # For RF-RC computational runtimes
# file = path + 'simulated_data/runtime' + str(n_features) + '_' + str(n_train) + 'rfrc_time'

# results_loaded = []
# with open(file, 'rb') as f:
#     results_loaded = pickle.load(f)

# all_reuslts = []

# for k, s0_ in enumerate(results_loaded):
#     all_reuslts.append(np.zeros([10, 6])) 
#     for i, row_ in enumerate(s0_):
#         for j, elem in enumerate(row_):
#             all_reuslts[k][i, j] = elem


#     # report data
    
#     print(f's0 = {k + 2}')
#     print(f'Runtime: {np.mean(all_reuslts[k], axis = 0)} \n'
#     f'--------------------')