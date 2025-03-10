#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:18:36 2025

@author: xbb
"""


from datetime import datetime
import warnings, os, pickle
from ohos.ProgressiveTree import ProgressiveTree, transform_features
# from method.dspione_v4 import DSPiOne_beta_v6, transform_features
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from ohos.util.param_search import search_start
from ohos.util.parameter_tuning_space import space_xgb, \
    objective_xgb_regression
    


import xgboost as xgb
xgb.__version__ # works with xgboost version 1.5.0
# $ conda install xgboost==1.5.0

from ohos.tree_functions import orf_train, porf_train, \
    xgb_train, rf_train, rf_plus_Sb_train

import warnings 
warnings.filterwarnings('ignore')

#%%


#%%

# parameter sets (s0, n_features)
hyper_ = [(2, 10), (2, 30), (2, 300)]



#%%
# Set 1
# Set parameters for the simulation
n_test = 5000      # Number of observations (rows)
n_train = 200 # 200
n_samples = n_test + n_train

noise_level = 1.0  # Noise level for the response variable
    
    
max_evals_ = 30

# Progressive tree
b_ = 1000
s_ = 6
n_S_ = 100
n_depth_ = 3


R = 100 # 100

save_ = False

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



for para_ in hyper_:

    n_features = para_[1]  # 10, 30, 300    # Number of covariates (columns)
    
    # s0 dimensional XOR
    s0 = para_[0]

    # Siulation results
    results = {}
    results['orf'] = []
    results['porf'] = []
    results['rf'] = []
    results['irf'] = []
    for r_ in range(R):
        
        #%%
        # generating data
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
        
        
        # three sample splitting
        X_train_in, X_train_out, y_train_in, y_train_out = \
            train_test_split(X_train, y_train, test_size=0.2)
            


        #%%
        
        start_time = time.time()
        orf_param, orf = orf_train(X_train_in, 
                                   X_train_out, 
                                   y_train_in, 
                                   y_train_out,
                                   max_evals = max_evals_)
        orf.fit(X_train, y_train)
        end_time = time.time()
        
        y_pred_dspione = orf.predict(X_test)
        mse_orf = mean_squared_error(y_test, y_pred_dspione)
        print(f"Mean Squared Error ORF: {mse_orf}, Use runtime: {end_time - start_time}")
        
        
        #%%
        start_time = time.time()
        porf_param, porf = porf_train(X_train_in, 
                                      X_train_out, 
                                      y_train_in, 
                                      y_train_out,
                                      max_evals = max_evals_)
        porf.fit(X_train, y_train)
        end_time = time.time()
        
        y_pred_dspione = porf.predict(X_test)
        mse_porf = mean_squared_error(y_test, y_pred_dspione)
        print(f"Mean Squared Error PORF: {mse_porf}, Use runtime: {end_time - start_time}")
        
        
        
        
        #%%
        start_time = time.time()
        rf_param, rf, _ = rf_train(X_train_in, 
                                   X_train_out, 
                                   y_train_in, 
                                   y_train_out,
                                   max_evals = max_evals_)
        rf.fit(X_train, y_train)
        end_time = time.time()
        
        y_pred_dspione = rf.predict(X_test)
        mse_rf = mean_squared_error(y_test, y_pred_dspione)
        print(f"Mean Squared Error RF: {mse_rf}, Use runtime: {end_time - start_time}")
        
        
        #%%
        # Initilize
        
        weights = []
        print(f'{r_}th experiment among {R} experiments.')
        print(f's0 = {s0}, s = {s_}, n_features = {n_features}', 
              f'n = {n_train}, b = {b_}, n_S = {n_S_}, noise_level = {noise_level}')
        
        start_time = time.time()
        irf, pro_tree = rf_plus_Sb_train(X_train_in, 
                                         X_train_out, 
                                         y_train_in, 
                                         y_train_out,
                                         b = b_,
                                         n_S = n_S_,
                                         s = s_,
                                         n_depth = n_depth_,
                                         max_evals = max_evals_)
        
        irf.fit(X_train, y_train)
        end_time = time.time()

        y_pred_dspione = irf.predict(X_test)
        mse_irf = mean_squared_error(y_test, y_pred_dspione)
        print(f"Mean Squared Error irf: {mse_irf}, Use runtime: {end_time - start_time}")
        
        
        
        #%%
        #####
        # Save the r square results for all models
        # results is a dict of 
        # r-square scores of each model
        results['orf'].append(1 - mse_orf / np.var(y_test))
        results['porf'].append(1 - mse_porf / np.var(y_test) )
        results['rf'].append(1 - mse_rf / np.var(y_test) )
        results['irf'].append(1 - mse_irf / np.var(y_test) )

        if save_:
            

            file = path + 'simulated_data/comparison' + str(n_features) + '_' + str(s0)
            with open(file, 'wb') as f:
                pickle.dump(results, f)
