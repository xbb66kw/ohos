#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:52:37 2025

@author: xbb
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:14:06 2023

@author: xbb
"""


# import time
import os, pickle
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np

# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.model_selection import train_test_split
# # import packages for hyperparameters tuning
# from hyperopt import Trials, fmin, tpe
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression

# from sklearn.ensemble import RandomForestRegressor
# # import xgboost as xgb
# # xgb.__version__ # works with xgboost version 1.5.0
# # $ conda install xgboost==1.5.0

# import openml



# from method.progressive_tree import progressive_tree, transform_features


# from method.util.param_search import search_start
# from method.util.parameter_tuning_space import space_xgb, \
#     objective_xgb_regression
    


# from method.tree_functions import orf_train2,  \
#     porf_train2, xgb_train,\
#         rf_train2, optimal_n_weight_train2



# #%%
# #####
# # Initialization
# run_irf = True
# run_ixgb = False
# run_orf = True
# run_porf = True
# run_xgb = False
# run_rf = True
# # save_ = True to save results
# save_ = False
# continued_ = False

# # whether considering interaction columns
# interaction_ = False
# # test version
# test_ = 'beta' # 'beta'

# # Do not touch these parameters
# max_evals = 100
# test_size = 0.5
# validation_size = 0.2

# # Progressive tree
# b_ = 200
# s_ = 6
# n_S_ = 100
# n_depth_ = 2

# # number of optimization evaluations
# max_evals_ = 100

# # Repeat R times. Default R = 10
# R = 10


# obj_rsquare_score = [[] for r in range(R)]


path_temp = os.getcwd()
result = path_temp.split("/")
path = ''
checker = True
for elem in result:
    if elem != 'dataset' and checker:
        path = path + elem + '/'
    else:
        checker = False
path = path + 'dataset' + '/'

print(path)
start_ind_sample = 0
start_repeatition = 0  


#%%
''' if __name__ == '__main__' is required for multiprocessing '''
if __name__ == '__main__':
    

    for ind_sample in range(start_ind_sample, 19):

        
        file = path + str(ind_sample)
        with open(file, 'rb') as f:
            dataset = pickle.load(f)


        X, y, categorical_indicator, attribute_names \
            = dataset.get_data(dataset_format = "dataframe", 
            target = dataset.default_target_attribute)
        
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        

        

        print(dataset)
        
        
        file = path + 'numpy' + str(ind_sample)
        print('Results for thebest parameters for openml are \
              saving at: ', '\n', file)
        # with open(file, 'wb') as f:
        #     pickle.dump((X, y, str(dataset)), f)
        #     pass
        
              
        
        