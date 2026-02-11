#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 13:45:57 2023

@author: xbb
"""
from pathlib import Path
import os, pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#%%


dataset_name = {
    0: "Ailerons",                # HD
    1: "superconduct",            # HD
    2: "yprop_4_1",               # HD
    3: "cpu_act",
    4: "pol",
    5: "elevators",
    6: "wine_quality",
    7: "houses",
    8: "house_16H",
    9: "diamonds",
    10: "Brazilian_houses",
    11: "Bike_Sharing_Demand",
    12: "nyc-taxi-green-dec-2016",
    13: "house\_sales",
    14: "sulfur",
    15: "medical_charges",
    16: "MiamiHousing2016",
    17: "abalone",
    18: "delay_zurich_transport"
}

# Metadata: Reordered to match dataset_name indices 0-18
dataset_meta = [
    {"name": "Ailerons", "total_p": 594},               # 0
    {"name": "superconduct", "total_p": 3239},          # 1
    {"name": "yprop\_4\_1", "total_p": 945},              # 2
    {"name": "cpu\_act", "total_p": 252},                # 3
    {"name": "pol", "total_p": 377},                    # 4
    {"name": "elevators", "total_p": 152},              # 5
    {"name": "wine\_quality", "total_p": 77},            # 6
    {"name": "houses", "total_p": 44},                  # 7
    {"name": "house\_16H", "total_p": 152},              # 8
    {"name": "diamonds", "total_p": 27},                # 9
    {"name": "Brazilian\_houses", "total_p": 44},        # 10
    {"name": "Bike\_Sharing\_Demand", "total_p": 27},     # 11
    {"name": "nyc-taxi-green-dec-2016", "total_p": 54}, # 12
    {"name": "house\_sales", "total_p": 135},            # 13
    {"name": "sulfur", "total_p": 27},                  # 14
    {"name": "medical\_charges", "total_p": 9},          # 15
    {"name": "MiamiHousing2016", "total_p": 104},       # 16
    {"name": "abalone", "total_p": 35},                 # 17
    {"name": "delay\_zurich\_transport", "total_p": 44}   # 18
]
# New ordered indices to match the dictionary above
all_indices = list(range(19))
# Subsets for your RRS calculations
hd_indices = [0, 1, 2]

#%% Load 

# Whether to use feature column interactions
interaction_ = True
# beta version simulation
test_ = 'beta' 

current_path = Path.cwd()
# Find 'ohos' in the hierarchy and set it as the base
# This works whether you are in ohos/, ohos/scripts/, or ohos/experiments/
base_path = next((p for p in [current_path] + list(current_path.parents) if p.name == 'ohos'), current_path)

# Ensure 'path' is a Path object for easy joining later
path = base_path


#####
# Manually control for outputing summary results
# Codes include file reading commends
rsquare_path = Path(path) / 'data' / 'openml' / 'results' / f"r_square{interaction_}{test_}"

with open(rsquare_path, 'rb') as f:
    obj_rsquare_score = pickle.load(f)

# Quick check on the result
print(f"Length of index 19: {len(obj_rsquare_score[19])}")


# Load Runtime results
runtime_path = Path(path) / 'data' / 'openml' / 'results' / f"runtime{interaction_}{test_}"

with open(runtime_path, 'rb') as f:
    obj_runtime = pickle.load(f)

# Display runtime object
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
    R = 20  # number of repetition in the numerical experiments
    D_ = 19 # 19
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
    method = 1
    print(f' {np.round(np.max(np.mean(score_all[method], axis=0)), 3)} & {np.round(np.mean(score_all[method]), 3)}  & {np.round(np.min(np.mean(score_all[method], axis=0)), 3)}') 
    
    print('average winning rate:',
          np.mean(score_all[method]), '\n',
          'max wining rate: ',
          np.max(np.mean(score_all[method], axis=0)), '\n',
          'min wining rate: ',
          np.min(np.mean(score_all[method], axis=0)))
    
    # maximum over 10 trails WOKRING!!!

    # Method names for the LaTeX table
    method_labels = [
        r"{\tiny RF+$\mathcal{S}^{(1000)}$}",
        r"{\tiny RF+$\mathcal{S}^{(2000)}$}",
        r"{\tiny RF+$\mathcal{S}^{(4000)}$}",
        r"{\tiny RF+$\mathcal{S}^{(8000)}$}",
        r"{\tiny F-RC}",
        r"{\tiny RF}",
        r"{\tiny MORF}"
    ]
    
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\scriptsize")
    print(r"    \begin{center}")
    print(r"        \begin{tabular}[t]{ |lccc| } \hline")
    print(r"            & maximum ${\textnormal{ALL-RRS}}$ & average ${\textnormal{ALL-RRS}}$ & minimum ${\textnormal{ALL-RRS}}$ \\")
    # print(r"            & $ \frac{\max_{r \in \mathcal{R}} \sum_{d\in \mathcal{D}}\textnormal{RRS}_{q, r, d} }{\texttt{\#} \mathcal{D} } $ & $\frac{(\texttt{\#} \mathcal{R})^{-1} \sum_{r \in \mathcal{R}} \sum_{d\in \mathcal{D}}\textnormal{RRS}_{q, r, d}} { \texttt{\#} \mathcal{D} }$ & $ \frac{\min_{r \in \mathcal{R}} \sum_{d\in \mathcal{D}}\textnormal{RRS}_{q, r, d} }{\texttt{\#} \mathcal{D} } $ \\ [0.5ex] \hline")
    
    for i in range(7):
        # Calculate the three statistics for the current method
        # axis=0 corresponds to the repetitions (R), j corresponds to datasets (D)
        # mean over j first, then find max/min/mean over repetitions
        rep_means = np.mean(score_all[i], axis=0)
        
        avg_rrs = np.mean(score_all[i])
        max_rrs = np.max(rep_means)
        min_rrs = np.min(rep_means)
        
        # Format the row
        row = f"            {method_labels[i]} & {max_rrs:.3f} & {avg_rrs:.3f} & {min_rrs:.3f} \\\\"
        print(row)
    
    print(r"            \hline")
    print(r"        \end{tabular}")
    print(r"        \caption{The table reports, from left to right, the maximum, average, and minimum of ALL-RRS scores across 20 independent evaluations for each method. } \label{tab:overall.a}")
    print(r"    \end{center}")
    print(r"\end{table}")

                                                                                                                
    #%%
    # Report the detailed R^2 scores
    # Method; Dataset; Repetition
    R = 20
    D_ = 19 # 19
    result_table = np.zeros(11 * D_ * R).reshape(11, D_, R)
    for j in range(D_):
        for ind in range(R):
            results = obj_rsquare_score[ind]
            result_table[:, j, ind] = [results[j]['obtree_1000'], 
                                       results[j]['obtree_2000'],
                                       results[j]['obtree_4000'],
                                       results[j]['obtree_8000'],                                       
                                       results[j]['irf_1000'],
                                       results[j]['irf_2000'],
                                       results[j]['irf_4000'],
                                       results[j]['irf_8000'],
                                       results[j]['orf'], 
                                       results[j]['rf'],
                                       results[j]['porf']]
    
    method_names = [
        "$PT^{(1000)}$", "$PT^{(2000)}$", 
        "$PT^{(4000)}$", "$PT^{(8000)}$",
        "RF+$\mathcal{S}^{(1000)}$", "RF+$\mathcal{S}^{(2000)}$", 
        "RF+$\mathcal{S}^{(4000)}$", "RF+$\mathcal{S}^{(8000)}$", 
        "F-RC", "RF", "MORF"
    ]
    
    
    
    
    
    # 3. Organize results into a structured list of dictionaries
    structured_results = []
    
    for j in range(min(len(dataset_meta), D_)):
        # Calculate stats once per dataset
        stats_per_method = {}
        for m_idx, m_name in enumerate(method_names):
            reps = result_table[m_idx, j, :]
            stats_per_method[m_name] = {
                "max": np.max(reps),
                "mean": np.mean(reps),
                "min": np.min(reps)
            }
        
        structured_results.append({
            "info": dataset_meta[j],
            "stats": stats_per_method
        })
        
        
    def generate_latex_table(data_list, methods, cols_per_row=4):
        output = []
        output.append(r"\begin{table}[ht!]\centering\scriptsize")
        output.append(r"\setlength{\tabcolsep}{2pt}")
        output.append(r"\resizebox{0.98\textwidth}{!}{")
        output.append(r"\begin{tabular}{|l|" + "c" * cols_per_row + "|}\hline")
    
        # Loop through datasets in batches of 4
        for i in range(0, len(data_list), cols_per_row):
            batch = data_list[i : i + cols_per_row]
            current_len = len(batch)
            
            # Header Row: Dataset Names
            header = "   "
            for item in batch:
                header += f"& {item['info']['name']} ({item['info']['total_p']}) "
            # Pad empty columns if the batch is smaller than cols_per_row
            header += " & " * (cols_per_row - current_len) + r"\\ \hline"
            output.append(header)
            
            # Method Rows: One row per method for the current batch of datasets
            for m_name in methods:
                row = f"            {m_name} "
                for item in batch:
                    s = item["stats"][m_name]
                    row += f"& ({s['max']:.3f}, {s['mean']:.3f}, {s['min']:.3f}) "
                row += " & " * (cols_per_row - current_len) + r"\\"
                output.append(row)
            output.append(r"\hline")
    
        output.append(r"\end{tabular}}")
        output.append(r"\caption{The values of (maximum, average, and minimum R$^2$ scores over 20 independent evaluations) are reported for each dataset and model. The numbers in parentheses following the dataset names represent the total number of features (see Table~\ref{tab:dataset_features_10rows})}\label{tab:r_square_updated_high}")
        output.append(r"\end{table}")
        
        return "\n".join(output)

    # Execution
    print(generate_latex_table(structured_results, method_names))
    #%%
    # Report the detailed runtime
    # Method; Dataset; Repetition
    R = 20
    D_ = 19 # 19
    result_table = np.zeros(11 * D_ * R).reshape(11, D_, R)
    for j in range(D_):
        for ind in range(R):
            results = obj_runtime[ind]
            result_table[:, j, ind] = [results[j]['obtree_1000'], 
                                       results[j]['obtree_2000'],
                                       results[j]['obtree_4000'],
                                       results[j]['obtree_8000'],                                       
                                       results[j]['irf_1000'],
                                       results[j]['irf_2000'],
                                       results[j]['irf_4000'],
                                       results[j]['irf_8000'],
                                       results[j]['orf'], 
                                       results[j]['rf'],
                                       results[j]['porf']]
    
    
    # Update labels to match the 11 rows above
    method_names = [
        "$PT^{(1000)}$", "$PT^{(2000)}$", 
        "$PT^{(4000)}$", "$PT^{(8000)}$",
        "RF+$\mathcal{S}^{(1000)}$", "RF+$\mathcal{S}^{(2000)}$", 
        "RF+$\mathcal{S}^{(4000)}$", "RF+$\mathcal{S}^{(8000)}$", 
        "F-RC", "RF", "MORF"
    ]
    
    
    # 3. Organize results into a structured list of dictionaries
    structured_results = []
    
    for j in range(min(len(dataset_meta), D_)):
        # Calculate stats once per dataset
        stats_per_method = {}
        for m_idx, m_name in enumerate(method_names):
            reps = result_table[m_idx, j, :]
            stats_per_method[m_name] = {
                "max": np.max(reps),
                "mean": np.mean(reps),
                "min": np.min(reps)
            }
        
        structured_results.append({
            "info": dataset_meta[j],
            "stats": stats_per_method
        })
        
        
    def generate_latex_table(data_list, methods, cols_per_row=4):
        output = []
        output.append(r"\begin{table}[ht!]\centering\scriptsize")
        output.append(r"\setlength{\tabcolsep}{2pt}")
        output.append(r"\resizebox{0.98\textwidth}{!}{")
        output.append(r"\begin{tabular}{|l|" + "c" * cols_per_row + "|}\hline")
    
        # Loop through datasets in batches of 4
        for i in range(0, len(data_list), cols_per_row):
            batch = data_list[i : i + cols_per_row]
            current_len = len(batch)
            
            # Header Row: Dataset Names
            header = "  "
            for item in batch:
                header += f"& {item['info']['name']} ({item['info']['total_p']}) "
            # Pad empty columns if the batch is smaller than cols_per_row
            header += " & " * (cols_per_row - current_len) + r"\\ \hline"
            output.append(header)
            
            # Method Rows: One row per method for the current batch of datasets
            for m_name in methods:
                row = f"            {m_name} "
                for item in batch:
                    s = item["stats"][m_name]
                    row += f"& ({s['max']:.1f}, {s['mean']:.1f}, {s['min']:.1f}) "
                row += " & " * (cols_per_row - current_len) + r"\\"
                output.append(row)
            output.append(r"\hline")
    
        output.append(r"\end{tabular}}")
        output.append(r"\caption{The maximum, average, and minimum computational runtimes (in seconds) over 20 independent evaluations are reported for each dataset and model. }\label{tab:runtime_comparison}")
        output.append(r"\end{table}")
        
        return "\n".join(output)

    # Execution
    print(generate_latex_table(structured_results, method_names))
    #%%
    # Print comparative scores based on high-dimenional applciations [4, 15, 16] d
    # Across datasets comparison
    # average distance to the minimum (ADTM)
    R = 20  # number of repetition in the numerical experiments
    D_ = len(hd_indices) # 3
    # Method; Dataset; Repetition
    result_table = np.zeros(7 * D_ * R).reshape(7, D_, R)
    for ind_j, j in enumerate(hd_indices):
        for ind in range(R):
            results = obj_rsquare_score[ind]
            result_table[:, ind_j, ind] = [results[j]['irf_1000'],
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

   
    

    # Method names for the LaTeX table
    method_labels = [
        r"{\tiny RF+$\mathcal{S}^{(1000)}$}",
        r"{\tiny RF+$\mathcal{S}^{(2000)}$}",
        r"{\tiny RF+$\mathcal{S}^{(4000)}$}",
        r"{\tiny RF+$\mathcal{S}^{(8000)}$}",
        r"{\tiny F-RC}",
        r"{\tiny RF}",
        r"{\tiny MORF}"
    ]
    
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\scriptsize")
    print(r"    \begin{center}")
    print(r"        \begin{tabular}[t]{ |lccc| } \hline")
    print(r"            & maximum ${\textnormal{HD-RRS}}$ & average ${\textnormal{HD-RRS}}$ & minimum ${\textnormal{HD-RRS}}$ \\")

    
    for i in range(7):
        # Calculate the three statistics for the current method
        # axis=0 corresponds to the repetitions (R), j corresponds to datasets (D)
        # mean over j first, then find max/min/mean over repetitions
        rep_means = np.mean(score_all[i], axis=0)
        
        avg_rrs = np.mean(score_all[i])
        max_rrs = np.max(rep_means)
        min_rrs = np.min(rep_means)
        
        # Format the row
        row = f"            {method_labels[i]} & {max_rrs:.3f} & {avg_rrs:.3f} & {min_rrs:.3f} \\\\"
        print(row)
    
    print(r"            \hline")
    print(r"        \end{tabular}")
    print(r"        \caption{The table reports, from left to right, the maximum, average, and minimum of HD-RRS scores across 20 independent evaluations for each method.  HD-RRS scores are calculated based on three high-dimensional datasets listed in Table~\ref{tab:dataset_features_10rows}.  } \label{tab:overall.b}")
    print(r"    \end{center}")
    print(r"\end{table}")
