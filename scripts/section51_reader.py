#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:09:26 2025

@author: xbb
"""

import warnings, os, pickle
import numpy as np
import math
#%%
n_features = 20 # 20
n_train = 250 # 250
            
noise_level = 0 # 0

hyper_ = [1, 2, 3, 4]

# Get the file path
path_temp = os.getcwd()
result = path_temp.split("/")
path = ''
checker = True
for elem in result:
    if elem != 'ohos' and checker:
        path = path + elem + '/'
    else:
        checker = False
path = path + 'ohos' + '/data/'

# all_reuslts = []
# file = path + 'simulated_data/runtime' + str(n_features) + '_'  + str(n_train)

# results_loaded = []
# with open(file, 'rb') as f:
#     results_loaded = pickle.load(f)

# # A R X 4 matrix
# # b in [2, 20, 200, 2000, 20000]
# # s0 in [2, 3, 4, 5]
# all_reuslts = np.zeros([50, 4, 5])
# for r_, current_results in enumerate(results_loaded):
#     all_reuslts[r_] = current_results


# results_loaded[0]
# len(results_loaded)

# # report data
# A = np.round(np.mean(all_reuslts, axis = 0), 2)
# B = np.round(np.std(all_reuslts, axis = 0), 2)


# for s0 in range(5):
#     print(" & ".join(f"{A[s0, i]} ({B[s0, i]})" for i in range(len(A[s0, :]))))

if False:
    #%%
    # oblique tree
    # max_f = 'paper_' 
    # max_f = 5000
    # max_f = 1000000
    max_f = None
    
    all_reuslts = []
    file = path + 'simulated_data/runtime' + str(n_features) + '_'  + str(n_train) + str(max_f) + 'obtree'
    
    
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    
    results_loaded = obj['metric']
    runtime_loaded = obj['runtime']
    
    # A R X 4 matrix
    # b in [2, 20, 200, 2000, 20000]
    # s0 in [1, 2, 3, 4]
    all_reuslts = np.zeros([50, 4, 5])
    for r_, current_results in enumerate(results_loaded):
        all_reuslts[r_] = current_results
    
    
    
    len(results_loaded)
    # 1. Process Runtime data
    # Use np.array to ensure we can perform math operations
    data_array = np.array(runtime_loaded)
    np.set_printoptions(suppress=True)
    
    # Calculate integer runtimes using ceiling
    runtime_int = np.round(np.mean(data_array, axis=0), 1)
    
    # 2. Process Metric data (R^2)
    A = np.mean(all_reuslts, axis=0)
    B = np.std(all_reuslts, axis=0)
    
    # 3. Set up B_vals with proper LaTeX formatting
    if max_f is None:
        B_vals = [r"$\infty$", r"$\infty$", r"$\infty$", r"$\infty$"]
    elif max_f == 'paper_':
        B_vals = [
            # r"{\tiny $2.8 \times 10^3$}", 
            # r"{\tiny $1.8 \times 10^5$}", 
            # r"{\tiny $3.1 \times 10^6$}", 
            # r"{\tiny $2.6 \times 10^7$}"
            r"{\tiny $3.4 \times 10^3$}", 
            r"{\tiny $3.2 \times 10^4$}", 
            r"{\tiny $1.9 \times 10^5$}", 
            r"{\tiny $8.2 \times 10^5$}"
        ]
    else:
        exponent = int(np.log10(max_f))
        coeff = max_f / (10**exponent)
        
        if coeff == 1.0:
            # Perfect power of 10 (e.g., 1000 -> 10^3)
            B_latex = fr"{{\tiny $10^{{{exponent}}}$}}"
        else:
            # Significant digit included (e.g., 5000 -> 5 \times 10^3)
            # Using :g to remove .0 if it is a clean integer
            B_latex = fr"{{\tiny ${coeff:g} \times 10^{{{exponent}}}$}}"
        
        B_vals = [B_latex] * 4
    
    iterations = [2, 20, 200, 2000, 20000]
    
    # 4. Generate Combined Data rows
    for i in range(len(A)):
        s0 = i + 1
        
        # format cells: 0.00(0.00)/0
        # :.2f ensures "0" becomes "0.00" for vertical alignment
        row_cells = [
            f"{A[i, j]:.2f}({B[i, j]:.2f})/{runtime_int[i, j]}" 
            for j in range(len(iterations))
        ]
        
        row_results = " & ".join(row_cells)
        
        # B_vals[i] already contains the tiny command and math $ signs
        print(f" & ${s0}$ & {B_vals[i]} & {row_results} \\\\")
    
    #%%
    # RF-RC
    
    
    
    file = path + 'simulated_data/runtime' + str(n_features) + '_' + str(n_train) + 'rfrc' + str(noise_level)
    
    
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    
    results_loaded = obj['metric']
    runtime_loaded = obj['runtime']
    len(results_loaded)
    # A R X 4 matrix
    
    
    # s0 in [1, 2, 3, 4]
    # max_features in [100, 1000, 10000, 100000, 1000000, 1000000]
    all_reuslts = np.zeros([50, 4, 6])
    for r_, current_results in enumerate(results_loaded):
        all_reuslts[r_] = current_results
    
    # 1. Process Metric Data (R^2)
    # Slice [:, 1:] to remove the first column (max_features = 100)
    A = np.mean(all_reuslts, axis=0)[:, 1:]
    B = np.std(all_reuslts, axis=0)[:, 1:]
    
    
    # 2. Process Runtime Data
    all_runtimes = np.array(runtime_loaded)
    # Take ceiling to get the integer runtimes (0, 1, etc.) shown in your example
    
    
    runtime_int = np.round(np.mean(all_runtimes, axis=0)[:, 1:], 2)
    
    # 3. Print Combined LaTeX Rows
    for i in range(A.shape[0]):
        s0 = i + 1
        
        # Format each cell: R^2(std)/runtime
        # :.2f ensures decimal alignment for the metrics
        row_cells = [
            f"{A[i, j]:.2f}({B[i, j]:.2f})/{runtime_int[i, j]}" 
            for j in range(A.shape[1])
        ]
        
        # Adding the & at the very beginning to accommodate the \multirow{4}{*}{Set} column
        print(f"{s0} & {' & '.join(row_cells)} \\\\")