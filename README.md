Install this Python package by following these steps:  

1. Download the **ohos** Python package.  
2. Open a terminal and navigate to the main folder of the **ohos** package.
3. Install required Python pacakges
   ```
   pip install numpy scikit-learn==1.6.1 treeple hyperopt
   ```
4. Run the following command:  
   ```
   pip install -e .
   ```  
5. Restart your Python editor if needed.

The `treeple` package is required for benchmark comparisons. The error "'RegressorTags' object has no attribute 'multi_label'" suggests that `treeple` might be incompatible with the version of scikit-learn you have installed. `treeple` likely relies on an older version of scikit-learn, where multi_label was present but has since been removed or changed.
   
