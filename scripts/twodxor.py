#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings, os, pickle
from ohos.ProgressiveTree import ProgressiveTree, transform_features
import numpy as np
import time

from treeple.ensemble import ObliqueRandomForestRegressor
from sklearn.model_selection import train_test_split
# Ensure mean_squared_error is imported as in your original snippet
from sklearn.metrics import mean_squared_error

from ohos.tree_functions import orf_train, porf_train, \
    rf_train, rf_plus_Sb_train

import warnings 
warnings.filterwarnings('ignore')

#%%
# parameter sets (s0, n_features)
hyper_ = [(2, 800, 400), (2, 400, 200)]

#%%
# Set parameters for the simulation
n_test = 5000      # Number of observations (rows)
noise_level = 1.0  
max_evals_ = 30

# Progressive tree parameters

b_ = 8000


s_ = 5
n_S_ = 100
n_depth_ = 3

R = 50
save_ = True

mse_orf = 0
mse_porf = 0
mse_rf = 0
mse_irf_high = 0
mse_brtree_high = 0
mse_obtree_infiniteB = 0
mse_obtree_finiteB = 0
                                  
                                  
for para_ in hyper_:
    n_features = para_[1]
    s0 = para_[0]
    
    n_train = para_[2] #  400 # 200, 400 
    n_samples = n_test + n_train
    
    # Simulation results for R-squared
    results = {key: [] for key in ['orf', 'porf', 'rf', 'irf_high', 'obtree_infiniteB', 'obtree_finiteB_2b', 'obtree_finiteB_3b', 'obtree_finiteB', 'brtree_high']}
    # Dictionary to record runtime for each method
    results_time = {key: [] for key in ['orf', 'porf', 'rf', 'irf_high', 'obtree_infiniteB', 'obtree_finiteB_2b', 'obtree_finiteB_3b', 'obtree_finiteB', 'brtree_high']}
    
    for r_ in range(R):
        print(f'{r_}th experiment among {R}. s0={s0}, n={n_train}, b={b_}, n_features = {n_features}')
        # generating data
        X = np.random.choice(a = [0, 1], 
                             size = n_samples * n_features,
                             replace = True).reshape(n_samples, n_features)
        
        index_set = np.random.choice(a = X.shape[1], size = s0, replace = False)
        output = np.ones(X.shape[0])
        output[np.sum(X[:, index_set], axis = 1) % 2 == 0] = -1
        y = output
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = n_test)
        y_train = y_train + np.random.randn(len(y_train)) * noise_level
        
        X_train_in, X_train_out, y_train_in, y_train_out = \
            train_test_split(X_train, y_train, test_size=0.2)

        # ORF
        start_time = time.time()
        orf_param, orf = orf_train(X_train_in, X_train_out, y_train_in, y_train_out, max_evals = max_evals_)
        orf.fit(X_train, y_train)
        y_pred_dspione = orf.predict(X_test)
        duration = time.time() - start_time
        results_time['orf'].append(duration)
        mse_orf = mean_squared_error(y_test, y_pred_dspione)
        print(f"MSE ORF: {mse_orf}, Runtime: {duration}")

       
        # RF
        start_time = time.time()
        rf_param, rf, _ = rf_train(X_train_in, X_train_out, y_train_in, y_train_out, max_evals = max_evals_)
        rf.fit(X_train, y_train)
        y_pred_dspione = rf.predict(X_test)
        duration = time.time() - start_time
        results_time['rf'].append(duration)
        mse_rf = mean_squared_error(y_test, y_pred_dspione)
        print(f"MSE RF: {mse_rf}, Runtime: {duration}")

 
        # IRF (Progressive Tree Transfer to RF)
        start_time = time.time()
        irf, pro_tree = rf_plus_Sb_train(X_train_in, X_train_out, y_train_in, y_train_out,
                                         b = b_, n_S = n_S_, s = s_, n_depth = n_depth_, max_evals = max_evals_)
        irf.fit(X_train, y_train)
        y_pred_dspione = irf.predict(X_test)
        duration = time.time() - start_time
        results_time['irf_high'].append(duration)
        mse_irf_high = mean_squared_error(y_test, y_pred_dspione)
        print(f"MSE high IRF: {mse_irf_high}, Runtime: {duration}")

        # OBTREE (Single Progressive Tree)
        start_time = time.time()
        ob_tree = ProgressiveTree(max_depth=3, s=5, min_samples_leaf=10, min_samples_split=10, n_S=100, paper_version=False)
        ob_tree.initialize(X_train, y_train)
        for _ in range(b_):
            ob_tree.restart(X_train, y_train)
        ob_tree.finalize(X_train, y_train)
        ob_tree.expanded_weights = ob_tree.used_weights
        y_pred_obtree = ob_tree.predict(X_test)        
        mse_obtree_infiniteB = mean_squared_error(y_test, y_pred_obtree)
        duration = time.time() - start_time
        results_time['obtree_infiniteB'].append(duration)
        print(f"MSE infinite B Oblique Tree: {mse_obtree_infiniteB}, Runtime: {duration}")
        
        
        
        # OBTREE (Single Progressive Tree, finite B)
        start_time = time.time()
        ob_tree = ProgressiveTree(max_depth=3, s=5, min_samples_leaf=10, min_samples_split=10, n_S=100, paper_version=True)
        ob_tree.B = 1000000
        ob_tree.initialize(X_train, y_train)
        for _ in range(b_):
            ob_tree.restart(X_train, y_train)
        ob_tree.finalize(X_train, y_train)
        ob_tree.expanded_weights = ob_tree.used_weights
        y_pred_obtree = ob_tree.predict(X_test)        
        mse_obtree_finiteB = mean_squared_error(y_test, y_pred_obtree)
        
        duration = time.time() - start_time
        results_time['obtree_finiteB'].append(duration)
        print(f"MSE finite B Oblique Tree: {mse_obtree_finiteB}, Runtime: {duration}")
        
        # Continued learning
        for _ in range(b_):
            ob_tree.restart(X_train, y_train)
        ob_tree.finalize(X_train, y_train)
        ob_tree.expanded_weights = ob_tree.used_weights
        y_pred_obtree = ob_tree.predict(X_test)        
        mse_obtree_finiteB_2b = mean_squared_error(y_test, y_pred_obtree)
        
        duration = time.time() - start_time
        results_time['obtree_finiteB_2b'].append(duration)
        print(f"MSE double finite B Oblique Tree: {mse_obtree_finiteB_2b}")
        
        
        # Continued learning
        for _ in range(b_):
            ob_tree.restart(X_train, y_train)
        ob_tree.finalize(X_train, y_train)
        ob_tree.expanded_weights = ob_tree.used_weights
        y_pred_obtree = ob_tree.predict(X_test)        
        mse_obtree_finiteB_3b = mean_squared_error(y_test, y_pred_obtree)
        
        duration = time.time() - start_time
        results_time['obtree_finiteB_3b'].append(duration)
        print(f"MSE tribple finite B Oblique Tree: {mse_obtree_finiteB_3b}")
        
        # BRTREE HIGH B
        start_time = time.time()
        orf_high = ObliqueRandomForestRegressor(n_estimators=1, max_features=1000000, max_depth=3,
                                                min_samples_leaf=10, min_samples_split=10,
                                                bootstrap=False, feature_combinations=s_)
        orf_high.fit(X_train, y_train)
        y_pred_high = orf_high.predict(X_test)
        duration = time.time() - start_time
        results_time['brtree_high'].append(duration)
        mse_brtree_high = mean_squared_error(y_test, y_pred_high)
        print(f"MSE Breiman High B: {mse_brtree_high}, Runtime: {duration}")

        

        # Record R-squared results
        y_var = np.var(y_test)
        results['orf'].append(1 - mse_orf / y_var)
        results['porf'].append(1 - mse_porf / y_var)
        results['rf'].append(1 - mse_rf / y_var)
        results['irf_high'].append(1 - mse_irf_high / y_var)
        results['obtree_infiniteB'].append(1 - mse_obtree_infiniteB / y_var)        
        results['obtree_finiteB'].append(1 - mse_obtree_finiteB / y_var)
        results['obtree_finiteB_2b'].append(1 - mse_obtree_finiteB_2b / y_var)        
        results['obtree_finiteB_3b'].append(1 - mse_obtree_finiteB_3b / y_var)        
        results['brtree_high'].append(1 - mse_brtree_high / y_var)

        # File path logic
        current = Path.cwd()

        # This finds the specific parent named 'o1Neuro' and appends 'data'
        # .parents includes all parent directories; .parts includes the current folder
        target_base = next((p for p in [current] + list(current.parents) if p.name == 'ohos'), current)

        path = target_base / 'data'

        if save_:
            # Use Path objects and f-strings for the most modern, clean approach
            file_path = Path(path) / 'simulated_data' / f"comparison{n_features}_{s0}"
            
            # Convert to string for functions that require it
            file = str(file_path)
            # You might want to save both results and results_time
            with open(file, 'wb') as f:
                pickle.dump({'metrics': results, 'runtime': results_time}, f)
