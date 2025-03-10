#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:56:34 2023

@author: xbb
"""
import numpy as np


def transform_features(X, weights_list, with_orthogonal = False):
    n_features = X.shape[1]
    if with_orthogonal:
        X_new = np.zeros([X.shape[0], n_features + len(weights_list)])
        for j in range(n_features):
            X_new[:, j] = X[:, j]
        for j, weights in enumerate(weights_list):
            X_new[:, j + n_features] = X @ weights
    else:
        X_new = np.zeros([X.shape[0], len(weights_list)])
        for j, weights in enumerate(weights_list):
            X_new[:, j] = X @ weights
    return X_new
    
    
    
