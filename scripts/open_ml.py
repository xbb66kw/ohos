#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:14:06 2023

@author: xbb
"""

from pathlib import Path
import time
import warnings, os, pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import openml



from treeple.ensemble import ObliqueRandomForestRegressor
from ohos.ProgressiveTree import ProgressiveTree
from ohos.tree_functions import orf_train, porf_train,\
        rf_train, rf_plus_Sb_train

# Install this package first before usage.
# cd [...]/ohos
# pip install -e .

###
### Restart the editor after the installation if needed.
###

#%%
#####
# Initialization
run_irf = True
run_orf = True
run_porf = True
run_rf = True
run_obtree = True
run_ottree = False
# save_ = True to save results
save_ = True


# whether considering interaction columns
interaction_ = True
# test version
test_ = 'beta' # 'beta' 'beta2' 'beta3'

# Do not touch these parameters
# test_size = 0.5
validation_size = 0.2

# Progressive tree
s_ = 5 # 5, 6
n_S_ = 100
n_depth_ = 3

# number of optimization evaluations
max_evals_ = 100

# Repeat R times. Default R = 10
R = 20


obj_rsquare_score = [[] for r in range(R)]
obj_runtime = [[] for r in range(R)]


current = Path.cwd()

# This finds the specific parent named 'o1Neuro' and appends 'data'
# .parents includes all parent directories; .parts includes the current folder
target_base = next((p for p in [current] + list(current.parents) if p.name == 'ohos'), current)

path = target_base / 'data'


start_ind_sample = 0
all_indices = [4, 15, 16, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18]
#%%
if __name__ == '__main__':
    # start_ind_sample = 0
    # run time is reported for house and superconduct datasets
    # for ind_sample in [8]:
    
    for ind_sample in all_indices:
    # for ind_sample in range(start_ind_sample, 19):
        file_path = Path(path) / 'openml' / 'dataset' / f"{ind_sample}"
        file = str(file_path)
        with open(file, 'rb') as f:        
            dataset = pickle.load(f)        
        X_, y_, dataset = dataset
        dataset_name = dataset[45:].split('\n')[0]
        
        
        for q in range(R):
            # Calculate the total samples needed: 400 for training + up to 2000 for testing
            # We ensure we don't exceed the actual available data in len(y_)
            max_test_needed = 2000
            train_fixed = 400
            total_needed = min(train_fixed + max_test_needed, len(y_))
            
            # Randomly sample the indices from the original dataset
            ind_rand = np.random.choice(np.arange(len(y_)), 
                                        size = total_needed, 
                                        replace = False)
            
            X = np.array(X_)[ind_rand, :]
            y = np.array(y_)[ind_rand]
            
            if interaction_:
                # Generates p original features + p*(p-1)/2 interactions + p squares
                # Total features = p*(p+1)/2 + p
                poly = PolynomialFeatures(degree=2, 
                                          interaction_only=False, 
                                          include_bias=False)
                X = poly.fit_transform(X)
            
            # Calculate the test_size ratio to ensure exactly 400 training samples
            # test_size here represents the proportion of the total_needed
            actual_test_size = (len(y) - train_fixed) / len(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=actual_test_size
            )
            
            print(f"Dataset: {dataset}")
            print(f"Training shape: {X_train.shape}") # Should be (400, features)
            print(f"Testing shape: {X_test.shape}")   # Should be (up to 2000, features)
            
            # Further split training for validation if necessary
            X_train_in, X_train_out, y_train_in, y_train_out = train_test_split(
                X_train, y_train, test_size = validation_size
            )
            # run irf
            r_square_orf = 0
            runtime_orf = 0
            if run_orf:                
                start_time = time.time()
                orf_param, orf = orf_train(X_train_in, 
                                           X_train_out, 
                                           y_train_in, 
                                           y_train_out,
                                           max_evals = max_evals_)
                orf.fit(X_train, y_train)
                end_time = time.time()
                
                y_pred_dspione = orf.predict(X_test)
                r_square_orf = 1 - mean_squared_error(orf.predict(X_test), y_test) / np.var(y_test)
                runtime_orf = end_time - start_time
                print(f"R square ORF: {r_square_orf}, Use runtime: {end_time - start_time}")
                

                
        
            #####
            # run porf
            r_square_porf = 0
            runtime_porf = 0
            if run_porf:                
                start_time = time.time()
                porf_param, porf = porf_train(X_train_in, 
                                              X_train_out, 
                                              y_train_in, 
                                              y_train_out,
                                              max_evals = max_evals_)
                porf.fit(X_train, y_train)
                end_time = time.time()
                
                y_pred_dspione = porf.predict(X_test)
                r_square_porf = 1 - mean_squared_error(porf.predict(X_test), y_test) / np.var(y_test)
                runtime_porf = end_time - start_time
                print(f"R square PORF: {r_square_porf}, Use runtime: {end_time - start_time}")


            #####
            # run Random Forests
            r_square_rf = 0
            runtime_rf = 0
            if run_rf:
                start_time = time.time()
                rf_param, rf, _ = rf_train(X_train_in, 
                                           X_train_out, 
                                           y_train_in, 
                                           y_train_out,
                                           max_evals = max_evals_)
                rf.fit(X_train, y_train)
                end_time = time.time()
                
                y_pred_dspione = rf.predict(X_test)
                r_square_rf = 1 - mean_squared_error(rf.predict(X_test), y_test) / np.var(y_test)
                runtime_rf = end_time - start_time
                print(f"R square RF: {r_square_rf}, Use runtime: {end_time - start_time}")
                
            #####
            # Run irf
            if run_irf:
                r_square_irf_ = []
                runtime_ = []
                pro_tree = None
                # 1000, 2000, 4000, 8000
                pro_tree = None
                B_ = 0
                current_runtime = 0
                s_last = 0
                for k, b_ in enumerate([1000, 1000, 2000, 2000, 2000]):
                    B_ += b_
                    
                    irf, pro_tree, s2, s4 = rf_plus_Sb_train(X_train_in, 
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
                    # end_time = time.time()
                    
                    y_pred_dspione = irf.predict(X_test)
                    r_square_irf_.append(1 - mean_squared_error(irf.predict(X_test), y_test) / np.var(y_test))
                    current_runtime = current_runtime + s2 - s_last + s4
                    runtime_.append(current_runtime)
                    s_last = s4
                    print(f"R square IRF{B_} : {r_square_irf_[k]}, Use runtime: {current_runtime}")
            else:
                r_square_irf_ = [0, 0, 0, 0, 0]
                runtime_ = [0, 0, 0, 0, 0]
            
            
            r_square_obtree_ = []
            runtime_obtree_ = []
            if run_obtree:
                start_time = time.time()
                # Initialize the final model with best parameters
                pro_tree = ProgressiveTree(
                    max_depth = 3, 
                    s = 5,
                    min_samples_leaf = 10,
                    min_samples_split = 10,
                    n_S = 100,
                    paper_version = False
                )
                # Set the max_features (B) parameter directly

                
                # We use the full training set (X_train_in) to settle on the 
                # final weights after finding the best hyperparameters.
                pro_tree.initialize(X_train, y_train)
                

                B_ = 0
                for k, b_ in enumerate([1000, 1000, 2000, 2000, 2000]):
                    B_ += b_
                    for _ in range(b_):
                        pro_tree.restart(X_train, y_train)
                    
                    pro_tree.finalize(X_train, y_train)
                    # Finalize the oblique weights for prediction
                    pro_tree.expanded_weights = pro_tree.used_weights
                    
                    end_time = time.time()
                    
                    y_pred_obtree = pro_tree.predict(X_test)
                    
                    r_square_obtree_.append(1 - mean_squared_error(y_pred_obtree, y_test) / np.var(y_test))
                    runtime_obtree_.append(end_time - start_time)
                    

    
                    print("R^2 of an oblique tree:", r_square_obtree_[k])
            else:
                r_square_obtree_ = [0, 0, 0, 0, 0]
                runtime_obtree_ = [0, 0, 0, 0, 0]
                
            r_square_ottree_ = 0
            runtime_ottree_ = 0
            if run_ottree:
                orf = ObliqueRandomForestRegressor(
                n_estimators=1, 
                max_features=min(X_train.shpae[1]^2, 2000),
                max_depth=3,
                min_samples_leaf=10,   # Minimum samples required to be at a leaf node
                min_samples_split=10,  # Minimum samples required to split an internal node
                bootstrap=False,           # Enables bagging/bootstrapping
                max_samples=None,
                feature_combinations = s_)
            
                start_time = time.time()  # Record start time
                orf.fit(X_train, y_train)
                end_time = time.time()  # Record end time
                elapsed_time = end_time - start_time  # Compute elapsed time
                
                
                
                y_pred_ottree = orf.predict(X_test)
                r_square_ottree_ = 1 - mean_squared_error(y_pred_ottree, y_test) / np.var(y_test)
                runtime_ottree_ = end_time - start_time
                
                
    
                print("R^2 of an Breiman's oblique tree:", r_square_ottree_)
            #####
            # Save file
            
            #####               
            obj_rsquare_score[q].append({
                'obtree_1000': r_square_obtree_[0],
                'obtree_2000': r_square_obtree_[1],
                'obtree_4000': r_square_obtree_[2],
                'obtree_6000': r_square_obtree_[3],
                'obtree_8000': r_square_obtree_[4],
                'irf_1000': r_square_irf_[0],
                'irf_2000': r_square_irf_[1],
                'irf_4000': r_square_irf_[2],
                'irf_6000': r_square_irf_[3],
                'irf_8000': r_square_irf_[4],
                'orf': r_square_orf,
                'rf': r_square_rf,
                'porf': r_square_porf,
                'dataset': dataset_name})
            
            obj_runtime[q].append({
                'obtree_1000': runtime_obtree_[0],
                'obtree_2000': runtime_obtree_[1],
                'obtree_4000': runtime_obtree_[2],
                'obtree_6000': runtime_obtree_[3],
                'obtree_8000': runtime_obtree_[4],
                'irf_1000': runtime_[0],
                'irf_2000': runtime_[1],
                'irf_4000': runtime_[2],
                'irf_6000': runtime_[3],
                'irf_8000': runtime_[4],
                'orf': runtime_orf,
                'rf': runtime_rf,
                'porf': runtime_porf,
                'dataset': dataset_name})
                    
            if save_:

                file_path = Path(path) / 'openml' / 'results' / f"r_square{interaction_}{test_}"

                # Ensure the results directory exists to avoid FileNotFoundError
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert to string if required by legacy functions, otherwise use file_path directly
                file = str(file_path)
                print('Results for thebest parameters for openml are \
                      saving at: ', '\n', file)
                with open(file, 'wb') as f:
                    pickle.dump(obj_rsquare_score, f)
                runtime_file_path = Path(path) / 'openml' / 'results' / f"runtime{interaction_}{test_}"

                # Ensure the 'results' directory exists
                runtime_file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(runtime_file_path, 'wb') as f:
                    pickle.dump(obj_runtime, f)
        start_repeatition = 0
                
              
        
        