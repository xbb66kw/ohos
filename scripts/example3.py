#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:27:50 2025

@author: xbb
"""





import numpy as np
import os, pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# pip install treeple
#   for using treeple pacakge
from treeple.ensemble import ObliqueRandomForestRegressor

# import xgboost as xgb
# xgb.__version__ # works with xgboost version 1.5.0
# $ conda install xgboost==1.5.0




import warnings 
warnings.filterwarnings('ignore')






# s0 dimensional XOR
hyper_ = [2, 3, 4, 5, 6] # the values of s0
hyper_2 = [100, 1000, 10000, 100000] # max_features


# Set 1
# Set parameters for the simulation
n_test = 5000      # Number of observations (rows)
n_train = 5000 # 500
n_samples = n_test + n_train
n_features = 20  # 10, 30, 300    # Number of covariates (columns)
noise_level = 1  # Noise level for the response variable

# For progressive tree
s_ = 6


save_ = False
restart_ = False

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


results = []
if save_ and not restart_:
    file = path + 'simulated_data/runtime' + str(n_features) + '_' + str(n_train) + 'rfrc'
    with open(file, 'rb') as f:
        results = pickle.load(f)

max_evals_ = 30
R_ = 100
run_start_ = len(results)
start_time_all = time.time()  # Record start time
for r_ in range(run_start_, R_):
    print(f'{r_}th round.')

    


    
    # Siulation results
    results.append([])
    for q_, para_ in enumerate(hyper_):
        # s0 dimensional XOR
        s0 = para_
        results[r_].append([])

        for max_features_ in hyper_2:
            X = np.random.choice(a = [0, 1], 
                             size = n_samples * n_features,
                             replace = True).reshape(n_samples, n_features)
        
            
            
            index_set = np.random.choice(a = X.shape[1], size = s0, replace = False)
            output = np.ones(X.shape[0])
            output[np.sum(X[:, index_set], axis = 1) % 2 == 0] = -1
            y = output
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size = n_test)
            
            y_train = y_train + np.random.randn(len(y_train)) * noise_level
            
            X_train_in, X_train_out, y_train_in, y_train_out = \
                train_test_split(X_train, y_train, test_size=0.2)
            
            #%%        
            print(f's0 = {s0}, s = {s_}, n_features = {n_features}', 
                  f'n = {n_train}, max_features = {max_features_}, noise_level = {noise_level}')
            
            orf = ObliqueRandomForestRegressor(
                n_estimators=100, 
                max_features=max_features_,
                max_depth=5,
                feature_combinations = s_)
            
            start_time = time.time()  # Record start time
            orf.fit(X_train, y_train)
            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Compute elapsed time
            
            
            
            y_pred_dspione = orf.predict(X_test)
            mse_orf = mean_squared_error(y_test, y_pred_dspione)
            print("Mean Squared Error ORF:", mse_orf)
            

            results[r_][q_].append(1 - mse_orf / np.var(y_test))
            
                #%%
            
            
    if save_:
        #####
        # Save the r square results for all models
        # results is a dict of 
        # r-square scores of each model


        file = path + 'simulated_data/runtime' + str(n_features) + '_' + str(n_train) + 'rfrc'
        with open(file, 'wb') as f:
            pickle.dump(results, f)

                    

end_time_all = time.time()  # Record start time
print(f'total runtime: {start_time_all - end_time_all}')


