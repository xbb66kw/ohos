#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:17:43 2025

@author: xbb
"""

import numpy as np


from hyperopt import hp
import time

from ohos.util.param_search import search_start
from ohos.util.parameter_tuning_space import \
    objective_orf_regression, space_orf,\
    objective_porf_regression, space_porf,\
    objective_rf_regression_aug, objective_rf_regression, space_rf,\
        p_s_porf, p_s_orf, p_s_rf,\
    objective_obtree_regression, space_obtree
        
    
# pip install treeple 
#   for using treeple package
# https://docs.neurodata.io/treeple/dev/index.html
from treeple.ensemble import ObliqueRandomForestRegressor, \
    PatchObliqueRandomForestRegressor
    
from ohos.CustomRF import CustomRF


from ohos.ProgressiveTree import ProgressiveTree

import warnings 
warnings.filterwarnings('ignore')


#%%

def orf_train(X_train_in, 
              X_train_out, 
              y_train_in,
              y_train_out, 
              max_evals = 30): 

    # 60%  sample for training and 40% sample for scoring the 
    # hyper-parameters.

        
    best_param_orf = search_start(
        X_train_in, 
        X_train_out, 
        y_train_in, 
        y_train_out,
        objective_orf_regression,
        space_orf,
        max_evals = max_evals
        )
    

    # The optimal set of hyperparameters
    best_param = best_param_orf
    if p_s_orf['feature_combinations'][best_param['feature_combinations']] is None:
        n_coms = None
    else:
        n_coms = min(p_s_orf['feature_combinations'][best_param['feature_combinations']], 
                     X_train_in.shape[1])
                     

    regr = ObliqueRandomForestRegressor(
        n_estimators=100,
        max_features=min((X_train_in.shape[1])**2, 1000),
        max_depth=p_s_orf['max_depth'][best_param['max_depth']],                                    
        min_impurity_decrease=p_s_orf['min_impurity_decrease'][best_param['min_impurity_decrease']],
        min_samples_leaf=int(best_param['min_samples_leaf']),
        min_samples_split=int(best_param['min_samples_split']),
        criterion=p_s_orf['criterion'][best_param['criterion']],
        bootstrap=p_s_orf['bootstrap'][best_param['bootstrap']],
        feature_combinations = n_coms)
    return best_param_orf, regr
#%%

def porf_train(X_train_in, 
                X_train_out, 
                y_train_in,
                y_train_out, 
                max_evals = 30): 
    best_param_porf = search_start(
        X_train_in, 
        X_train_out, 
        y_train_in, 
        y_train_out,
        objective_porf_regression,
        space_porf,
        max_evals = max_evals
        )
    


    best_param = best_param_porf
    H1 = min(best_param['max_patch_dims'], X_train_in.shape[1])
    H2 = min(best_param['min_patch_dims'], X_train_in.shape[1])
    h1 = max(H1, H2)
    h2 = min(H1, H2)
    
    pregr = PatchObliqueRandomForestRegressor(
        n_estimators=100, 
        max_features=min((X_train_in.shape[1])**2, 1000),
        max_depth=p_s_porf['max_depth'][best_param['max_depth']],                                    
        min_impurity_decrease=p_s_porf['min_impurity_decrease'][best_param['min_impurity_decrease']],
        min_samples_leaf=int(best_param['min_samples_leaf']),
        min_samples_split=int(best_param['min_samples_split']),
        criterion=p_s_porf['criterion'][best_param['criterion']],
        bootstrap=p_s_porf['bootstrap'][best_param['bootstrap']],
        max_patch_dims = [1, h1],
        min_patch_dims = [1, h2])
    
    return best_param_porf, pregr

#%%
def rf_train(X_train_in, 
              X_train_out, 
              y_train_in,
              y_train_out, 
              pro_tree = None, 
              max_evals = 30): 
    
    if pro_tree is not None:
        space_rf['pro_tree'] = pro_tree
        
        # 0: oblique splits are not used
        # 1: oblique splits at the first layer are used
        # 2^{k} - 1: oblique splits before the kth layer are used
        if pro_tree.max_depth < 3:
            p_s_n_weights = [0]
            for k in range(pro_tree.max_depth):
                p_s_n_weights.append(max(2**(k + 1) - 1, 0))
        else:    
            p_s_n_weights = [0, 1, 3, 7, float('Inf')]
        space_rf['n_weights'] = hp.choice('n_weights', p_s_n_weights)
        
    
        best_param_rf = search_start(
            X_train_in, 
            X_train_out, 
            y_train_in, 
            y_train_out,
            objective_rf_regression_aug,
            space_rf,
            max_evals = max_evals
            )
    else:
        best_param_rf = search_start(
            X_train_in, 
            X_train_out, 
            y_train_in, 
            y_train_out,
            objective_rf_regression,
            space_rf,
            max_evals = max_evals
            )
        best_param_rf['n_weights'] = 0
        p_s_n_weights = [None]
    
    best_param = best_param_rf
    rf = CustomRF(
        n_estimators=100, 
        max_features=best_param['gamma'],
        max_depth=p_s_rf['max_depth'][best_param['max_depth']],                                    
        min_impurity_decrease=p_s_rf['min_impurity_decrease'][best_param['min_impurity_decrease']],
        min_samples_leaf=int(best_param['min_samples_leaf']),
        min_samples_split=int(best_param['min_samples_split']),
        criterion=p_s_rf['criterion'][best_param['criterion']])
    # rf model is needed to be arbitrarily trained first 
    # before further optimizing the number of weights.
    # Seem to be a bug. 
    rf.fit(X_train_in, y_train_in)
    
    return best_param_rf, rf, p_s_n_weights[best_param['n_weights']]


    #%%
def optimal_n_weight_train(X_train_in, 
                           y_train_in,                            
                           b: int = 1000, 
                           n_S: int = 100, 
                           s: int = 5,
                           n_depth: int = 3,
                           max_evals: int = 30, 
                           pro_tree = None):
    '''
    

    Parameters
    ----------
    X_train_in : TYPE
        DESCRIPTION.
    y_train_in : TYPE
        DESCRIPTION.
    b : int, optional
        Number of progressive tree iterations. The default is 1000.
    n_S : int, optional
        Number of available oblique splits for each iteration. 
        The default is 500.
    s : int, optional
        Number of nonzero elements in the oblique splits. The default is 1.
    n_depth : int, optional
        oblique tree depth. The default is 3.
    max_evals : int, optional
        DESCRIPTION. The default is 30.
    pro_tree : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    pro_tree : TYPE
        DESCRIPTION.

    '''
    # If pro_tree is provided, it is used 
    # instead of initializing a new pro_tree    
    if pro_tree is None:
        # Initialize a new progressive tree
        pro_tree = ProgressiveTree(max_depth = n_depth, 
                                   s = s, 
                                   n_S = n_S)    
        pro_tree.initialize(X_train_in, y_train_in)
    else:
        pro_tree_temp = ProgressiveTree()
        pro_tree_temp.copy_tree(pro_tree)
        pro_tree = pro_tree_temp
    
    for _ in range(b):
        pro_tree.restart(X_train_in, y_train_in)
            

    return pro_tree

#%%

def rf_plus_Sb_train(X_train_in,
                     X_train_out, 
                     y_train_in, 
                     y_train_out,
                     b: int = 1000,
                     n_S: int = 100,
                     s: int = 5,
                     n_depth: int = 3,
                     max_evals: int = 30,
                     pro_tree = None
                     ):
    '''
    

    Parameters
    ----------
    X_train_in : ndarray
        Input predictor features for training.
    X_train_out : ndarray
        Input predictor features for validation.
    y_train_in : ndarray
        Input response features for training.
    y_train_out : ndarray
        Input response features for validation.
    b : int, optional
        Number of iterations of the progressive tree. The default is 1000.
    n_S : int, optional
        Number of oblique splits to be optimize at each
        node. The default is 100.
    s : int, optional
        Number of nonzero coordinates in the oblique splits. 
        The default is 6.
    n_depth : int, optional
        Depth of the progressive tree. The default is 3.
    max_evals : int, optional
        Number of trials of hyperparamter optimization. 
        The default is 30.
    pro_tree : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    irf : TYPE
        A CustomRF object.
    pro_tree : TYPE
        A ProgressiveTree object.

    '''
    s1 = time.time()
    # pro_tree is a ProgressiveTree object.
    pro_tree = optimal_n_weight_train(X_train_in, 
                                      y_train_in, 
                                      b = b,
                                      n_S = n_S,
                                      s = s,
                                      n_depth = n_depth,                                          
                                      pro_tree = pro_tree)
    s2 = time.time() - s1
    # irf is a CustomRF object.
    s3 = time.time()
    irf_param, irf, n_w_irf = rf_train(X_train_in, 
                                       X_train_out, 
                                       y_train_in, 
                                       y_train_out, 
                                       pro_tree, 
                                       max_evals = max_evals)
    s4 = time.time() - s3
    used_weights = [weights for i, weights in 
            enumerate(pro_tree.used_weights) if i < n_w_irf]
    
    irf.expanded_weights = used_weights

    return irf, pro_tree, s2, s4