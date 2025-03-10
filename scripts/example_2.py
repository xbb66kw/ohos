#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:57:55 2025

@author: xbb
"""




from ohos.ProgressiveTree import transform_features
import numpy as np
import os, pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



from ohos.tree_functions import rf_plus_Sb_train

# import xgboost as xgb
# xgb.__version__ # works with xgboost version 1.5.0
# $ conda install xgboost==1.5.0




import warnings 
warnings.filterwarnings('ignore')


# s0 dimensional XOR
hyper_ = [2,3,4,5,6] # the values of s0


# Set 1
# Set parameters for the simulation
n_test = 5000      # Number of observations (rows)
n_train = 500 # 500
n_samples = n_test + n_train
n_features = 20  # 10, 30, 300    # Number of covariates (columns)
noise_level = 1  # Noise level for the response variable

# For progressive tree
s_ = 6
n_S_ = 100 
n_depth_ = 3 


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
    #####
    # Save the r square results for all models
    # results is a dict of 
    # r-square scores of each model

    file = path + 'simulated_data/runtime' + str(n_features) + '_' + str(n_train)
    with open(file, 'rb') as f:
        results = pickle.load(f)
    

max_evals_ = 30
R_ = 100
run_start_ = len(results)
for r_ in range(run_start_, R_):

    print(f'{r_}th round.')

    # s0 dimensional XOR
    
    
    # Siulation results
    results.append([])
    for q_, para_ in enumerate(hyper_):
        s0 = para_
        
        results[r_].append([])
        #%%
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

        time_start = time.time()
        pro_tree = None
        # [2, 20, 200, 2000, 20000]
        B_ = 0
        for b_ in [2, 18, 180, 1800, 18000]:
            B_ += b_
            
            print(f'{B_}th iteration starts.')
            print(f's0 = {s0}, s = {s_}, n_features = {n_features}', 
                  f'n = {n_train}, b = {B_}, n_S = {n_S_}, noise_level = {noise_level}')
            
            irf, pro_tree = rf_plus_Sb_train(X_train_in, 
                                             X_train_out, 
                                             y_train_in, 
                                             y_train_out,
                                             b = b_, # No. iterations
                                             n_S = n_S_, # Available splits
                                             s = s_, # No. nonzero 
                                             n_depth = n_depth_,
                                             max_evals = max_evals_,
                                             pro_tree = pro_tree)
            
            
            irf.fit(X_train, y_train)
            y_pred_dspione = irf.predict(X_test)
            mse_irf = mean_squared_error(y_test, y_pred_dspione)
            print("Mean Squared Error:", mse_irf)
            results[r_][q_].append(1 - mse_irf / np.var(y_test))

        time_end = time.time()
        print(f'Runtime : {time_start - time_end}, runtime per iteration {(time_start - time_end) / b_}')
            #%%
        
        
    if save_:
        #####
        # Save the r square results for all models
        # results is a dict of 
        # r-square scores of each model


        file = path + 'simulated_data/runtime' + str(n_features) + '_' + str(n_train)
        with open(file, 'wb') as f:
            pickle.dump(results, f)

