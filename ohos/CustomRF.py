#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:10:43 2025

@author: xbb
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ohos.util.transform_features import transform_features


    
class CustomRF(RandomForestRegressor):
    def fit(self, X, y):
        if hasattr(self, 'expanded_weights') \
            and self.expanded_weights is not None:
            X_test_aug = transform_features(
                X, 
                self.expanded_weights,
                with_orthogonal = True
                )
            super().fit(X_test_aug, y)  # Call original fit method


        else:
            super().fit(X, y)  # Call original fit method
    def predict(self, X):        
        if hasattr(self, 'expanded_weights') \
            and self.expanded_weights is not None:
            X_test_aug = transform_features(
                X, 
                self.expanded_weights,
                with_orthogonal = True
                )

            return np.mean([tree.predict(X_test_aug) for tree in self.estimators_], axis = 0)
        else:
            return np.mean([tree.predict(X) for tree in self.estimators_], axis = 0)


