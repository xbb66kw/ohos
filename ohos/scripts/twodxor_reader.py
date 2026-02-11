# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:30:50 2025

@author: xbb
"""

import warnings, os, pickle
import numpy as np


hyper_ = [400]
# hyper_ = [800]

s0 = 2

# Get the file path
path_temp = os.getcwd()
result = path_temp.split("/")
path = ''
checker = True
for elem in result:
    if elem != 'ohos' and checker:
        path = path + elem + '/'
    else:
        checker = False
path = path + 'ohos' + '/data/'



# Assuming s0, path, and hyper_ are already defined
for para_ in hyper_:
    n_features = para_[1] if isinstance(para_, tuple) else para_
    file = path + 'simulated_data/comparison' + str(n_features) + '_' + str(s0)
    
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            
        res = data['metrics'] if 'metrics' in data else data
        rtime = data['runtime'] if 'runtime' in data else data
        order = ['obtree_finiteB', 'obtree_finiteB_2b', 'obtree_finiteB_3b', 'obtree_infiniteB', 'brtree_high', 'irf_high', 'rf', 'orf']
        
        # Process R2 and Runtime into list strings
        r2_row = [f"{np.mean(res[n]):.2f} ({np.std(res[n]):.2f})" if n in res else "---" for n in order]
        rt_row = [f"{np.mean(rtime[n]):.2f} ({np.std(rtime[n]):.2f})" if n in rtime else "---" for n in order]

        # Output rows using \multirow for the p-value
        print(f"\\multirow{{2}}{{*}}{{{n_features}/{int(n_features/2)}}} & $R^2$ & {' & '.join(r2_row)} \\\\")
        print(f" & t & {' & '.join(rt_row)} \\\\")
        print(" \\hline")
        
    except FileNotFoundError:
        print(f"% File not found for n_features = {n_features}")

