Install this Python package by following these steps:  

0. Require Python (>=3.9).
1. Download the **ohos** Python package.
2. Rename the folder to **ohos** (default name: **ohos-main**).
3. Open a terminal and navigate to the main folder of the **ohos** package.
4. Install required Python pacakges
   ```
   pip install numpy pandas scikit-learn==1.6.1 treeple==0.9.1 hyperopt
   ```
5. Run the following command to install the **ohos** package:
   ```
   pip install -e .
   ```
6. You can check within Python by trying to import the package:
   ```
   try:
      import ohos
      print("Package is installed.")
   except ImportError:
      print("Package is NOT installed.")
   ```
8. Restart your Python editor if needed.

If you are unable to install this package, you can still use the provided source files, starting with `ohos/script/example.py`. If you only need the source code for my progressive tree, the `treeple` package is not required. However, `treeple` is necessary for benchmark comparisons.
   
