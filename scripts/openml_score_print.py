#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 13:45:57 2023

@author: xbb
"""

import os, pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#%%


dataset_name = {0: "cpu_act",
                1: "pol",
                2: "elevators",
                3: "wine_quality",
                4: "Ailerons",
                5: "houses",
                6: "house_16H",
                7: "diamonds",
                8: "Brazilian_houses",
                9: "Bike_Sharing_Demand",
                10: "nyc-taxi-green-dec-2016",
                11: "house_sales",
                12: "sulfur",
                13: "medical_charges",
                14: "MiamiHousing2016",
                15: "superconduct",
                16: "yprop_4_1",
                17: "abalone",
                18: "delay_zurich_transport"}

#%% Load 

# Whether to use feature column interactions
interaction_ = True
# beta version simulation
test_ = 'beta_runtime' # 'beta' 'beta_runtime'




# Get the directory path for loading data_process_embryogrowth.rds
path_temp = os.getcwd()
result = path_temp.split("/")

path = ''
checker = True
for elem in result:
    if elem != 'ohos' and checker:
        path = path + elem + '/'
    else:
        checker = False
path = path + 'ohos' + '/'
# My path is '/Users/xbb/Dropbox/', where 'xbb' is the name of 
# my device.

#####
# Manually control for outputing summary results
# Codes include file reading commends

file = path + 'data/openml/results/r_square' + str(interaction_) + str(test_)
with open(file, 'rb') as f:
    obj_rsquare_score = pickle.load(f)
len(obj_rsquare_score[9])
# obj_rsquare_score[0][18]

file = path + 'data/openml/results/runtime' + str(interaction_) + str(test_)
with open(file, 'rb') as f:
    obj_runtime = pickle.load(f)
obj_runtime

#%%
#####
# obj_rsquare_score is a list of length 10. Each records the 
# R^2 scores for all four methods (including the linear 
# regression) on each of the 19 datasets.
# See obj_rsquare_score[j], j = 0, ..., 18 for details.
if False:
    #%%
    #####
    # Across datasets comparison
    # average distance to the minimum (ADTM)
    R = 10  # number of repetition in the numerical experiments
    D_ = 19
    # Method; Dataset; Repetition
    result_table = np.zeros(7 * D_ * R).reshape(7, D_, R)
    for j in range(D_):
        for ind in range(R):
            results = obj_rsquare_score[ind]
            result_table[:, j, ind] = [results[j]['irf_1000'],
                                       results[j]['irf_2000'],
                                       results[j]['irf_4000'],
                                       results[j]['irf_8000'],
                                       results[j]['orf'], 
                                       results[j]['rf'],
                                       results[j]['porf']]

    score_all = np.zeros(7 * D_ * R).reshape(7, D_, R)
    for j in range(D_):
        for ind in range(R):
            M = np.max(result_table[:, j, ind])
            m = np.min(result_table[:, j, ind])
            for method in range(7):
                # Win rates
                score_all[method, j, ind] = \
                    (result_table[method, j, ind] - m) / (M - m)

    # Print the overall results
    # method = 0 (irf_1000), 1 (irf_2000), 2 (irf_4000)
    # 3 (irf_8000), 4 (orf), 5 (rf), 6 (porf)
    method = 0
    print(f' {np.round(np.max(np.mean(score_all[method], axis=0)), 3)} & {np.round(np.mean(score_all[method]), 3)}  & {np.round(np.min(np.mean(score_all[method], axis=0)), 3)}') 
    
    print('average winning rate:',
          np.mean(score_all[method]), '\n',
          'max wining rate: ',
          np.max(np.mean(score_all[method], axis=0)), '\n',
          'min wining rate: ',
          np.min(np.mean(score_all[method], axis=0)))
    #%%
    # Report the detailed R^2 scores
    # Method; Dataset; Repetition
    result_table = np.zeros(4 * D_ * R).reshape(4, D_, R)
    for j in range(D_):
        for ind in range(R):
            results = obj_rsquare_score[ind]
            result_table[:, j, ind] = [results[j]['irf_1000'],
                                       results[j]['orf'], 
                                       results[j]['rf'],
                                       results[j]['porf']]

    # Print the results
    # ind_dataset = 0, ..., 18
    ind_dataset = 0
    print('The R^2 scores for all three methods (irf, orf, rf, porf) based\
          on the ' + str(ind_dataset) + 'th dataset: ',
          dataset_name[ind_dataset])

    print('max:', np.max(result_table, axis=2)[:, ind_dataset])
    print('mean', np.mean(result_table, axis=2)[:, ind_dataset])
    print('min: ', np.min(result_table, axis=2)[:, ind_dataset])
    
    for ind_dataset in range(19):
        a = np.max(result_table, axis=2)[:, ind_dataset]
        b = np.mean(result_table, axis=2)[:, ind_dataset]
        c = np.min(result_table, axis=2)[:, ind_dataset]
        print(f'{dataset_name[ind_dataset]} \n {np.column_stack((a, b, c))}')
#%%
# Print runtime
# two datasets: house and superconditicity
    R = 10
    D_ = 2 # 2
    runtime_set = [5, 15]
    result_table = np.zeros(6 * D_ * R).reshape(6, D_, R)
    for j in range(D_):
        for ind in range(R):
            results = obj_runtime[ind]
            result_table[:, j, ind] = [results[j]['irf_1000'],
                                       results[j]['irf_2000'],
                                       results[j]['irf_4000'],
                                       results[j]['irf_8000'],
                                       results[j]['orf'], 
                                       results[j]['rf']]



    for ind_dataset in range(D_):
        a = np.round(np.max(result_table, axis=2)[:, ind_dataset])
        b = np.round(np.mean(result_table, axis=2)[:, ind_dataset])
        c = np.round(np.min(result_table, axis=2)[:, ind_dataset])
        print(f'{dataset_name[runtime_set[ind_dataset]]} \n {np.column_stack((a, b, c))}')

#%%
# print numbers of features
    path_temp = os.getcwd()
    result = path_temp.split("/")
    path = ''
    checker = True
    for elem in result:
        if elem != 'ohot' and checker:
            path = path + elem + '/'
        else:
            checker = False
    path = path + 'ohot' + '/'
    
        
    for ind_sample in range(19):
        file = path + 'data/openml/dataset/numpy' + str(ind_sample)
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        X_, y_, dataset = dataset
        if interaction_:
            poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
            X_ = poly.fit_transform(X_)
        dataset_name_ = dataset[45:].split('\n')[0]
        print(X_.shape, dataset_name_)
            

