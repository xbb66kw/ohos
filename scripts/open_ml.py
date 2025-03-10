#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:14:06 2023

@author: xbb
"""


import time
import warnings, os, pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import openml





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
# save_ = True to save results
save_ = False
continued_ = False

# whether considering interaction columns
interaction_ = False
# test version
test_ = 'beta' # 'beta' or 'beta_runtime'

# Do not touch these parameters
test_size = 0.5
validation_size = 0.2

# Progressive tree
s_ = 6
n_S_ = 100
n_depth_ = 3

# number of optimization evaluations
max_evals_ = 100

# Repeat R times. Default R = 10
R = 10


obj_rsquare_score = [[] for r in range(R)]
obj_runtime = [[] for r in range(R)]

path_temp = os.getcwd()
result = path_temp.split("/")
path = ''
checker = True
for elem in result:
    if elem != 'ohos' and checker:
        path = path + elem + '/'
    else:
        checker = False
path = path + 'ohos/data' + '/'


start_ind_sample = 0
start_repeatition = 0  
if continued_:
    file_ = path + 'data/openml/results/r_square' + str(interaction_) + str(test_)
    print(f' Files saved at {file_}')
    with open(file_, 'rb') as f:
        obj_rsquare_score = pickle.load(f)
    
    
    
    content_length = []
    for content in obj_rsquare_score:
        content_length.append(len(content))
    if all(content_length - np.mean(content_length) == 0):
        start_ind_sample = content_length[0]
    elif any(content_length - np.mean(content_length) < 0):
        start_ind_sample = content_length[0] - 1
        start_repeatition = np.where(content_length - np.mean(content_length) < 0)[0][0]


#%%
if __name__ == '__main__':
    # start_ind_sample = 0
    # run time is reported for house and superconduct datasets
    # for ind_sample in [5, 15]:
    for ind_sample in range(start_ind_sample, 19):
        file = path + 'openml/dataset/numpy' + str(ind_sample)
        with open(file, 'rb') as f:        
            dataset = pickle.load(f)        
        X_, y_, dataset = dataset
        dataset_name = dataset[45:].split('\n')[0]
        
        
        for q in range(start_repeatition, R):
            ind_rand = np.random.choice(np.arange(len(y_)), 
                    size = min(800, len(y_)), replace = False)
            X = np.array(X_)[ind_rand, :]
            y = np.array(y_)[ind_rand]
            
            if interaction_:
                poly = PolynomialFeatures(degree=2, 
                                          interaction_only=False, 
                                          include_bias=False)
                X = poly.fit_transform(X)
            
            
            X_train, X_test, y_train, y_test\
                = train_test_split(X, y, test_size=test_size)
            
            
            print(X_train.shape)
            print(dataset)
            
            X_train_in, X_train_out, y_train_in, y_train_out \
                = train_test_split(X_train, y_train, 
                                   test_size = validation_size)
        
            #####
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

            # run xgb
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
                for k, b_ in enumerate([1000, 1000, 2000, 4000]):
                    B_ += b_
                    start_time = time.time()
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
                    end_time = time.time()
                    
                    y_pred_dspione = irf.predict(X_test)
                    r_square_irf_.append(1 - mean_squared_error(irf.predict(X_test), y_test) / np.var(y_test))
                    runtime_.append(end_time - start_time)
                    print(f"R square IRF{B_} : {r_square_irf_[k]}, Use runtime: {end_time - start_time}")

                
            #####
            # Save file
            
            #####               
            obj_rsquare_score[q].append({
                'irf_1000': r_square_irf_[0],
                'irf_2000': r_square_irf_[1],
                'irf_4000': r_square_irf_[2],
                'irf_8000': r_square_irf_[3],
                'orf': r_square_orf,
                'rf': r_square_rf,
                'porf': r_square_porf,
                'dataset': dataset_name})
            
            obj_runtime[q].append({
                'irf_1000': runtime_[0],
                'irf_2000': runtime_[1],
                'irf_4000': runtime_[2],
                'irf_8000': runtime_[3],
                'orf': runtime_orf,
                'rf': runtime_rf,
                'porf': runtime_porf,
                'dataset': dataset_name})
                    
            if save_:
                file = path + 'openml/results/r_square' + str(interaction_) + str(test_)
                print('Results for thebest parameters for openml are \
                      saving at: ', '\n', file)
                with open(file, 'wb') as f:
                    pickle.dump(obj_rsquare_score, f)
                file = path + 'openml/results/runtime' + str(interaction_) + str(test_)
                with open(file, 'wb') as f:
                    pickle.dump(obj_runtime, f)
        start_repeatition = 0
                
              
        
        