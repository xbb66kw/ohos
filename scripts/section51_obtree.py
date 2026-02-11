#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 16:43:36 2026

@author: xbb
"""


from pathlib import Path
import math
from ohos.ProgressiveTree import transform_features
import numpy as np
import os, pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


from ohos.ProgressiveTree import ProgressiveTree


# Install this package first before usage.
# cd [...]/ohos
# pip install -e .




import warnings 
warnings.filterwarnings('ignore')


# s0 dimensional XOR
hyper_ = [1, 2, 3, 4] # the values of s0
# 

# Set 1
# Set parameters for the simulation
n_test = 5000      # Number of observations (rows)
n_train = 250 # 150, 250
n_samples = n_test + n_train
n_features = 20  # 20    # Number of covariates (columns)
noise_level = 0  # Noise level for the response variable
pre_B = 1000000 # 1000000, 5000, None

B = None

# For progressive tree
n_S_ = 100
save_ = True

current_path = Path.cwd()

# This finds the 'ohos' directory in the current hierarchy and appends 'data'
# It works whether you are in ohos/, ohos/scripts/, or ohos/experiments/
base_path = next((p for p in [current_path] + list(current_path.parents) if p.name == 'ohos'), current_path)
path = base_path / 'data'

# Ensure the folder exists (optional but very helpful)
path.mkdir(parents=True, exist_ok=True)


results = {}
results['metric'] = []
results['runtime'] = []
# results = []


max_evals_ = 30
R_ = 50

for r_ in range(R_):

    print(f'{r_}th round.')

    # s0 dimensional XOR
    
    
    # Siulation results
    results['metric'].append([])
    results['runtime'].append([])

    for q_, para_ in enumerate(hyper_):
        s0 = para_
        
        results['metric'][r_].append([])
        results['runtime'][r_].append([])
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
        # [2, 20, 200, 2000, 20000]
        B_ = 0
        for b_ in [2, 18, 180, 1800, 18000]:
            B_ += b_
            
            
            if pre_B == 1000000:
                B = 1000000
            elif pre_B == 5000:
                B = 5000
            else:
                log_term = (np.log(n_train)) ** 3
                comb_term = math.comb(n_features, s0)
                B = int(log_term * comb_term)

            
            print(f'{B_}th iteration starts.')
            print(f's0 = {s0}, n_features = {n_features}', 
                  f'n = {n_train}, b = {B_}, B (max_features) = {B}, noise_level = {noise_level}')
            
            if B is None:
                paper_version = False
            else:
                paper_version = True
            # Initialize the final model with best parameters
            pro_tree = ProgressiveTree(
                max_depth = 3, 
                s = 5,
                min_samples_leaf = 10,
                min_samples_split = 10,
                n_S = 100,
                paper_version = paper_version
            )
            # Set the max_features (B) parameter directly
            pro_tree.B = B
            
            # We use the full training set (X_train_in) to settle on the 
            # final weights after finding the best hyperparameters.
            pro_tree.initialize(X_train, y_train)
            

            for _ in range(B_):
                pro_tree.restart(X_train, y_train)
                
            pro_tree.finalize(X_train, y_train)
            # Finalize the oblique weights for prediction
            pro_tree.expanded_weights = pro_tree.used_weights
            
            y_pred_obtree = pro_tree.predict(X_test)
            

            mse_obtree = mean_squared_error(y_test, y_pred_obtree)

            print("Mean Squared Error of an oblique tree:", mse_obtree)
          
            results['metric'][r_][q_].append(1 - mse_obtree / np.var(y_test))
            
            time_end = time.time()
            results['runtime'][r_][q_].append(time_end - time_start)
            

        # time_end = time.time()
        print(f'Runtime : {time_end - time_start}, runtime per iteration {(time_start - time_end) / b_}')
            #%%
        

    if save_:
        #####
        # Save the r square results for all models
        if B is None:
            suffix = 'None'
        elif B in {1000000, 5000}:
            suffix = str(B)
        else:
            suffix = 'paper_'
        
            
        
        file_path = Path(path) / 'simulated_data' / f"runtime{n_features}_{n_train}{suffix}obtree"

        # Convert back to string if your legacy code requires it
        file = str(file_path)
        
        with open(file, 'wb') as f:
            pickle.dump(results, f)

            # file = path + 'simulated_data/runtime' + str(n_features) + '_' + str(n_train)  + 'paper_' + 'obtree'
            
