#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 19:45:07 2023

@author: xbb
"""
from typing import Any, Dict
import numpy as np
# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, hp
from sklearn.metrics import mean_squared_error
from hyperopt.pyll.base import scope
# Useful when debugging
import hyperopt.pyll.stochastic
# print(hyperopt.pyll.stochastic.sample(space))
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
xgb.__version__ # works with xgboost version 1.5.0
# $ conda install xgboost==1.5.0



from treeple.ensemble import ObliqueRandomForestRegressor,\
    PatchObliqueRandomForestRegressor

from ohos.ProgressiveTree import transform_features

#%%
# Tuning parameter space and objective functions for XGBoost

space_xgb = {
    'max_depth': scope.int(hp.quniform("max_depth", 2, 15, 1)),
    'gamma': hp.uniform('gamma', np.log(1e-8), np.log(7)),
    'reg_alpha': hp.uniform('reg_alpha', np.log(1e-8), np.log(1e2)),
    'reg_lambda': hp.uniform('reg_lambda', np.log(0.8), np.log(4)),
    'learning_rate': hp.uniform('learning_rate', np.log(1e-5), np.log(0.7)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 20, 1),
    'n_estimators': 1000}

# print(hyperopt.pyll.stochastic.sample(space_xgb))
# test_set = hyperopt.pyll.stochastic.sample(space_xgb)
# print(np.exp(test_set['gamma']), np.exp(test_set['learning_rate']), np.exp(test_set['reg_alpha']), np.exp(test_set['reg_lambda']), test_set['min_child_weight'])


def objective_xgb_regression(space):


    X_train_in, X_train_out, y_train_in, y_train_out = space['data']
        
    model=xgb.XGBRegressor(
        n_estimators =space['n_estimators'], 
        max_depth = int(space['max_depth']), 
        gamma = np.exp(space['gamma']),
        reg_alpha = np.exp(space['reg_alpha']),
        reg_lambda = np.exp(space['reg_lambda']),
        learning_rate = np.exp(space['learning_rate']),
        min_child_weight=space['min_child_weight'],
        colsample_bytree=space['colsample_bytree'],
        colsample_bylevel = space['colsample_bylevel'],
        subsample = space['subsample'])
   

    
    # Define evaluation datasets.
    evaluation = [( X_train_in, y_train_in), 
                  ( X_train_out, y_train_out)]
    
    # Fit the model. Define evaluation sets, early_stopping_rounds,
    # and eval_metric.
    model.fit(X_train_in, y_train_in,
            eval_set=evaluation, eval_metric="rmse",
            early_stopping_rounds=20,verbose=False)

    # Obtain prediction and rmse score.
    pred = model.predict(X_train_out)
    rmse = mean_squared_error(y_train_out, pred)
    
    # Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK, 'model': model}


#%%
def objective_n_weights_regression(space):


    X_train_in, X_train_out, y_train_in, y_train_out = space['data']
    

    pro_tree = space['pro_tree']
    n_weights = space['n_weights']
    used_weights = [weights for i, weights in 
            enumerate(pro_tree.used_weights) if i < n_weights]

    X_train_in = transform_features(X_train_in, 
                                     used_weights,
                                     with_orthogonal = True)
    X_train_out = transform_features(X_train_out, 
                                     used_weights,
                                     with_orthogonal = True)
    model = space['model']
    # Fit the model. Define evaluation sets, early_stopping_rounds,
    # and eval_metric.
    model.fit(X_train_in, y_train_in)

    # Obtain prediction and rmse score.
    pred = model.predict(X_train_out)
    rmse = mean_squared_error(y_train_out, pred)
    
    # Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK, 'model': model}
    
#%%
p_s_rf = {'max_depth': [None, 5, 10, 20, 50],
          'min_impurity_decrease': [0, 0.01, 0.02, 0.05],
          'criterion': ['squared_error', 'absolute_error']}

space_rf = {
    'gamma': hp.uniform('gamma', 0, 1), 
    'max_depth': hp.choice('max_depth', p_s_rf['max_depth']),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_impurity_decrease': hp.choice('min_impurity_decrease', p_s_rf['min_impurity_decrease']),
    'criterion': hp.choice('criterion', p_s_rf['criterion'])}


def objective_rf_regression_aug(space):

    X_train_in, X_train_out, y_train_in, y_train_out = space['data']
    # 60%  sample for training and 40% sample for scoring the 
    # hyper-parameters.
    # Do not touch these parameters 
    # unless you know what you're doing.

    pro_tree = space['pro_tree']
    n_weights = space['n_weights']
    used_weights = [weights for i, weights in 
            enumerate(pro_tree.used_weights) if i < n_weights]

    X_train_in = transform_features(X_train_in, 
                                     used_weights,
                                     with_orthogonal = True)
    X_train_out = transform_features(X_train_out, 
                                     used_weights,
                                     with_orthogonal = True)
    
    clf = RandomForestRegressor(
        n_estimators=100,
        max_depth = space['max_depth'],
        max_features = space['gamma'],
        min_samples_leaf = int(space['min_samples_leaf']),
        min_samples_split = int(space['min_samples_split']),
        criterion = space['criterion'],
        min_impurity_decrease = space['min_impurity_decrease']
        )
    clf.fit(X_train_in, y_train_in)   
    prediction = clf.predict(X_train_out)
    rmse = mean_squared_error(y_train_out, prediction)
    
    #Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK}

def objective_rf_regression(space):

    X_train_in, X_train_out, y_train_in, y_train_out = space['data']
    # 60%  sample for training and 40% sample for scoring the 
    # hyper-parameters.
    # Do not touch these parameters 
    # unless you know what you're doing.


    clf = RandomForestRegressor(
        n_estimators=100,
        max_depth = space['max_depth'],
        max_features = space['gamma'],
        min_samples_leaf = int(space['min_samples_leaf']),
        min_samples_split = int(space['min_samples_split']),
        criterion = space['criterion'],
        min_impurity_decrease = space['min_impurity_decrease']
        )
    clf.fit(X_train_in, y_train_in)   
    prediction = clf.predict(X_train_out)
    rmse = mean_squared_error(y_train_out, prediction)
    
    #Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK}


#%%
p_s_orf = {'max_depth': [None, 5, 10, 20, 50],
          'min_impurity_decrease': [0, 0.01, 0.02, 0.05],
          'criterion': ['squared_error', 'absolute_error'],
          'bootstrap': [True, False],
          'feature_combinations': [1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 20, None],
          'max_features': [None, 'sqrt', 'log2', 10, 100, 1000]}

space_orf = {'max_depth': hp.choice('max_depth', p_s_orf['max_depth']),
              'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 20, 1),
              'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
              'min_impurity_decrease': hp.choice('min_impurity_decrease', p_s_orf['min_impurity_decrease']),
              'criterion': hp.choice('criterion', p_s_orf['criterion']),
              'bootstrap': hp.choice('bootstrap', p_s_orf['bootstrap']),
              'feature_combinations': hp.choice('feature_combinations', p_s_orf['feature_combinations'])}

def objective_orf_regression(space):

    X_train_in, X_train_out, y_train_in, y_train_out = space['data']
    
        
    if space['feature_combinations'] is None:
        n_coms = None
    else:
        n_coms = min(space['feature_combinations'], X_train_in.shape[1])
    
    clf = ObliqueRandomForestRegressor(max_depth = space['max_depth'],
                max_features = min((X_train_in.shape[1])**2, 1000),
                min_samples_leaf = int(space['min_samples_leaf']),
                min_samples_split = int(space['min_samples_split']),
                criterion = space['criterion'],
                min_impurity_decrease = space['min_impurity_decrease'],
                bootstrap = space['bootstrap'],
                feature_combinations = n_coms)
    clf.fit(X_train_in, y_train_in)   
    prediction = clf.predict(X_train_out)
    rmse = mean_squared_error(y_train_out, prediction)
    
    #Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK}
#%%
p_s_porf = {'max_depth': [None, 5, 10, 20, 50],
          'min_impurity_decrease': [0, 0.01, 0.02, 0.05],
          'criterion': ['squared_error', 'absolute_error'],
          'bootstrap': [True, False]}

space_porf = {
    'max_depth': hp.choice('max_depth', p_s_orf['max_depth']),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_impurity_decrease': hp.choice('min_impurity_decrease', p_s_porf['min_impurity_decrease']),
    'criterion': hp.choice('criterion', p_s_porf['criterion']),
    'bootstrap': hp.choice('bootstrap', p_s_porf['bootstrap']),
    'max_patch_dims': hp.quniform('max_patch_dims', 1, 15, 1),
    'min_patch_dims': hp.quniform('min_patch_dims', 1, 15, 1)}



def objective_porf_regression(space):

    X_train_in, X_train_out, y_train_in, y_train_out = space['data']

    
    

    

    H1 = min(space['max_patch_dims'], X_train_in.shape[1])
    H2 = min(space['min_patch_dims'], X_train_in.shape[1])
    h1 = max(H1, H2)
    h2 = min(H1, H2)
    
    clf = PatchObliqueRandomForestRegressor(max_depth = space['max_depth'],
                max_features = min((X_train_in.shape[1])**2, 1000),
                min_samples_leaf = int(space['min_samples_leaf']),
                min_samples_split = int(space['min_samples_split']),
                criterion = space['criterion'],
                min_impurity_decrease = space['min_impurity_decrease'],
                bootstrap = space['bootstrap'],
                max_patch_dims = [1, h1],
                min_patch_dims = [1, h2])
    clf.fit(X_train_in, y_train_in)   
    prediction = clf.predict(X_train_out)
    rmse = mean_squared_error(y_train_out, prediction)
    
    #Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK}



