#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:30:50 2025

@author: xbb
"""

import warnings, os, pickle
import numpy as np


hyper_ = [10, 30, 300]
s0 = 2

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
    n_features = para_
    file = path + 'simulated_data/comparison' + str(n_features) + '_' + str(s0)
    
    results_loaded = {}
    with open(file, 'rb') as f:
        results_loaded = pickle.load(f)
    
    results_loaded
    for name_ in ['orf', 'porf', 'rf', 'irf']:
        print(f'n_features = {para_} Method {name_}: Mean = {np.round(np.mean(results_loaded[name_]), 2)}', 
              f' std = {np.round(np.std(results_loaded[name_]), 2)}.')


results_loaded['rf']
