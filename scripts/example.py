#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 15:14:44 2025

@author: xbb
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from ohos.tree_functions import rf_plus_Sb_train

import warnings 
warnings.filterwarnings('ignore')


#%%
# For the data-generating model
n_features = 100
# Since we are considering simulated data, 
# we may generate an arbitrarily large test sample.
n_test = 2000
n_train = 200
n_samples = n_train + n_test
noise_level = 1
s0 = 2 # s0-dimensional XOR problem
#%% 
# For the progressive tree
n_depth_ = 1 # progressive tree depth
s_ = 6 # s-sparse oblique splits
n_S_ = 100 # number of oblique splits for optimization at each node.
b_ = 50 # number of iterations
#%%
# For hyperparameter optimization
max_evals_ = 30 # number of trials of hyperparameter optimization
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
# Initilize
# pro_tree is a ProgressiveTree object.
# irf is a CustomRF object.
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

y_pred_dspione = irf.predict(X_test)
mse_irf = mean_squared_error(y_test, y_pred_dspione)
print(f"R^2 score of our progressive tree: { 1 - mse_irf / np.var(y_test)}.")
        
        
#%%
# Continue training the progressive tree model
irf, pro_tree = rf_plus_Sb_train(X_train_in, 
                                 X_train_out, 
                                 y_train_in, 
                                 y_train_out,
                                 b = b_,
                                 n_S = n_S_,
                                 s = s_,
                                 n_depth = n_depth_,
                                 pro_tree = pro_tree,
                                 max_evals = max_evals_)
irf.fit(X_train, y_train)

y_pred_dspione = irf.predict(X_test)
mse_irf = mean_squared_error(y_test, y_pred_dspione)

print(f"R^2 score: { 1 - mse_irf / np.var(y_test)}.")
