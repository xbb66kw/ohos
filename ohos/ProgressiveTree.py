#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 21:01:41 2025

@author: xbb
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 23:25:40 2024

@author: xbb
"""



# from typing import Any
import numpy as np
from ohos.node import Node
from ohos.util.transform_features import transform_features


class ProgressiveTree():
    def __init__(self, 
                 min_samples_split: int = 1,
                 min_samples_leaf: int = 1, 
                 max_depth: float = 3,
                 s: int = 5,
                 n_S: int = 100,
                 paper_version: bool = False
                 ) -> None:
       
    

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.s = s
        self.n_S = n_S
        
        self.paper_version = paper_version
        self.B = 1

            
        # initially not use any oblique splits
        self.expanded_weights = None
    def copy_tree(self, progressive_tree):
        self.used_weights = progressive_tree.used_weights
        self.min_samples_split = progressive_tree.min_samples_split
        self.min_samples_leaf = progressive_tree.min_samples_leaf
        self.max_depth = progressive_tree.max_depth
        self.s = progressive_tree.s
        self.n_S = progressive_tree.n_S
    def initialize(self, X: np.ndarray, y: np.ndarray):
        
        self.min_samples_split = min(self.min_samples_split,
                                     int(X.shape[0] / 10))
        self.X0 = X.copy()
        self.y_mean = np.mean(y)
        self.y = y - self.y_mean
        self.used_weights = []
        self.current_weights = []
        
        self.pre_sampled = []
        
        
        
        if self.paper_version:
            p = self.X0.shape[1]
            
            
            
            B = self.B
            a_ = min(p, self.s)
            
            # 1. Pre-generate all normal values in one go
            weight_matrix = np.random.normal(0, 1, (B, p))
            
            # 2. Create a sparsity mask
            # We generate a random matrix and use partition to pick top 'size' indices per row
            # This is much faster than np.random.choice in a loop
            mask = np.zeros((B, p), dtype=bool)
            random_rank = np.random.rand(B, p)
            
            for i in range(B):
                size_ = np.random.randint(1, a_ + 1)
                # Efficiently get indices of the 'size_' largest random values
                # equivalent to choosing 'size_' indices at random
                idx = np.argpartition(random_rank[i], -size_)[-size_:]
                mask[i, idx] = True
                
            # 3. Apply mask and Normalize
            weight_matrix[~mask] = 0
            norms = np.linalg.norm(weight_matrix, axis=1, keepdims=True)
            
            # Handle rows that might be all zeros (though unlikely with normal distribution)
            norms[norms == 0] = 1
            weight_matrix /= norms
            
            # 4. Store as list of arrays to match your update_current_weights logic
            self.pre_sampled = list(weight_matrix)

        self.restart(self.X0, self.y)
    def fit(self) -> None:
        '''
        

        Parameters
        ----------
        X : np.ndarray            
        y : np.ndarray

        Returns
        -------
        None
            Creates a forest of decision trees using a random 
            subset of data and features.

        '''
        y = self.y
        X = self.X
        n_samples = y.shape[0]
        

        # Initialization
        self.node_v = [Node(X, 
                            None, # feature_index
                            None, # threshold
                            np.arange(n_samples), # ind_set
                            0, # depth
                            0, # node_index
                            0 # predicted value
                            )]

        # Only the nodes with indices in end_node_index would be 
        # considered to be updated
        end_node_index = [0]
        # Predicted values
        prediction_y_mat = np.zeros(n_samples)

        # Loop until all nodes are sufficiently small or not valid 
        # for further splitting
        while len(end_node_index):
        
        
            node = self.node_v[end_node_index[0]]
            node_, daughter_node_list = \
                self.grow(X, y, prediction_y_mat, node)

            end_node_index.remove(node_.node_index)

            for daughter_node in daughter_node_list:
                # Valid nodes are appended into `end_node_index`
                if len(daughter_node.ind_set) > \
                    self.min_samples_split and \
                        daughter_node.depth < self.max_depth\
                            and not daughter_node.stop:
                    end_node_index.append(\
                        daughter_node.node_index)

                # Upate the node vector
                self.node_v.append(daughter_node)
                
                # Update the predicted values
                prediction_y_mat[daughter_node.ind_set] = daughter_node.ave

    def grow(self, 
             X: np.ndarray,
             y: np.ndarray,
             prediction_y_mat: np.ndarray,
             node
             ) -> tuple[list[Node], list[Node]]:

        # Calculate the residuals used for updating nodes
        y_residual_0 = y - prediction_y_mat
        y_residual_s = y_residual_0**2

 
            
        j_loss = node.cal_loss(X, 
                               y, 
                               y_residual_0, 
                               y_residual_s)

        
        ind_best = np.argmax(j_loss)

        
        # Record the potential daughter nodes for the 
        # updating nodes.
        node_daughter_list = []   

        # Release the memory
        del node.X_arg_temp
        del node.seq1
        del node.seq2
        
        
        # `update_ind_list` is a list of indices of the subsample
        update_ind_list = []
        ind_set = node.ind_set
        

        
        j = (ind_best // (len(ind_set) - 1)) 
        X_j = X[ind_set, :][:, j]
        sorted_indices = np.argsort(X_j)
        i = ind_best % (len(ind_set) - 1)
        threshold = (X_j[sorted_indices[i]]\
                      + X_j[sorted_indices[i+1]]) / 2
        
        ind_subset = X_j <= threshold
        # Xj with smaller values first
        update_ind_list.append(ind_subset)
        update_ind_list.append(~ind_subset)




        y_temp = y[ind_set]
        


        node.threshold = threshold
        node.feature_index = j 
        
        # Record the used weight 
        # These used split weights are the output selective 
        # oblique splits in mathcal{S}^{(b)}
        if node.depth + 1 <= self.max_depth:
            self.used_weights.append(self.current_weights[j])
        
        for counter_, ind_subset in enumerate(update_ind_list):
            # Small nodes do not continue to grow

            false_ave = np.mean(y_temp[ind_subset])
            child_node = Node(
                X,
                None, # feature_index
                None, # threshold
                node.ind_set[ind_subset], # subsample indices
                node.depth + 1, # depth
                len(self.node_v) + counter_, # node index
                false_ave # predicted values
                )
            child_node.parent = node

            if sum(ind_subset) <= self.min_samples_leaf:
                child_node.stop = True
                
            # Append to node's child_node list
            node.children.append(child_node)
            # Append to daughter node list                                         
            node_daughter_list.append(child_node)
            
        return node, node_daughter_list
    def restart(self, X, y):
        # WORKING!!!
        # Remove X, y input here?
        # Do I have to reset self.X0 and self.y_mean, self.y?
        self.X0 = X
        self.y_mean = np.mean(y)
        self.y = y - self.y_mean
        
        self.update_current_weights()
        p_= len(self.current_weights)
        self.X = np.zeros([self.X0.shape[0], p_])
        for j in range(p_):
            self.X[:, j] = self.X0 @ self.current_weights[j]

        # Train the tree model
        self.fit()
    def finalize(self, X, y):
        self.X0 = X
        self.y_mean = np.mean(y)
        self.y = y - self.y_mean
        
        # Initialize current_weights
        self.current_weights = []
        
        # Include p orthogonal (axis-aligned) splits:
        # These are identity vectors [1, 0, 0], [0, 1, 0], etc.
        n_features = self.X0.shape[1]
        for i in range(n_features):
            vector = np.zeros(n_features)
            vector[i] = 1.0
            self.current_weights.append(vector)
        
        # Extend with the best oblique weights found during Progressive rounds
        self.current_weights.extend(self.used_weights)
        
        # 4. Prepare the projected data X
        total_projections = len(self.current_weights)
        n_samples = self.X0.shape[0]
        self.X = np.zeros([n_samples, total_projections])
        
        for j in range(total_projections):
            self.X[:, j] = self.X0 @ self.current_weights[j]

        # Train the final tree model on the combined feature space
        self.fit()
    def update_current_weights(self):
        a_ = min(self.X0.shape[1], self.s)
        self.current_weights = []

        # Include additional n_weights_exploration exploration weights
        for _ in range(self.n_S):
            if self.paper_version:
                # 1. Check if we have pre-sampled weights available
                if not hasattr(self, 'pre_sampled') or len(self.pre_sampled) == 0:
                    raise ValueError("pre_sampled set is empty or not initialized.")
                
                # 2. Randomly pick one weight from the pre-sampled set
                # We use an index to select the weight vector
                idx = np.random.randint(0, len(self.pre_sampled))
                weights = self.pre_sampled[idx]
            else:
                # Passive learning
                size_ = np.random.choice(a = a_, size = 1)[0] + 1
                nonzero_ind = np.random.choice(a = self.X0.shape[1], 
                                               size = size_,
                                               replace = False)
                vec = np.random.normal(0, 1, size_)
                vec1 = np.zeros(self.X0.shape[1])
                vec1[nonzero_ind] = vec
                weights = vec1
                
                # Normalize
                weights /= np.linalg.norm(weights, ord = 2)
            
            self.current_weights.append(weights)

        self.current_weights.extend(self.used_weights)
        # Reinitialize the set of used weights
        self.used_weights = []
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.expanded_weights is not None:
            X_test_aug = transform_features(
                X, 
                self.expanded_weights,
                with_orthogonal = True
                )
            return self.node_v[0].predict(X_test_aug) + self.y_mean
        else:
            return self.node_v[0].predict(X) + self.y_mean
