import os

import numpy as np
import pandas as pd
from pingouin import multivariate_normality
from scipy.stats import kstest

# Iterate over all datasets in the folder
root_dir = "../results/exp_cv_2/"

for_classes = True

datasets_summary = pd.DataFrame(columns=["dataset_id", "n attributes", "n instances", "mvn_2", "mvn_5", "p_gaussian"])
for i,dataset_id in enumerate(os.listdir(root_dir)):
    if dataset_id == "plots" or dataset_id == "plots_analysis":
        continue
    dataset_path = os.path.join(root_dir, dataset_id)
    if os.path.isdir(dataset_path):
        file = os.path.join(dataset_path, "resampleddata_" + dataset_id + ".csv")
        df = pd.read_csv(file, index_col=0)
        datasets_summary.loc[i] = [dataset_id, df.shape[1] - 1, df.shape[0], True, True, None]
        #df = df.head(10000)
        if for_classes:
            # Gaussian count. Number of features that are conditionally Gaussian given the class
            gc = np.zeros(shape=(len(df[df.columns[-1]].unique()), df.shape[1] - 1))
            # Check if the dataset is MVN using the pingouin test for all class values
            for j,class_value in enumerate(df[df.columns[-1]].unique()):
                df_class = df[df[df.columns[-1]] == class_value]
                mvn = False, False,False # multivariate_normality(df_class.drop(df.columns[-1], axis=1), alpha=0.05)
                mvn_5 = False, False, False #multivariate_normality(df_class.drop(df.columns[-1], axis=1), alpha=1 - .9999994)
                for n_feature in range(df.shape[1] - 1):
                    # Test if the feature is conditionally Gaussian given the class
                    gc[j, n_feature] = kstest(df_class[df_class.columns[n_feature]], 'norm')[1] > 1 - .9999994
            datasets_summary.loc[i, "mvn_2"] = mvn[2] and datasets_summary.loc[i, "mvn_2"]
            datasets_summary.loc[i, "mvn_5"] = mvn_5[2] and datasets_summary.loc[i, "mvn_5"]
            gc = gc.max(axis=0)
            gc = gc.sum()
            datasets_summary.loc[i, "p_gaussian"] = gc
        else :
            mvn = False, False,False #multivariate_normality(df.drop(df.columns[-1], axis=1), alpha=0.05)
            mvn_5 = multivariate_normality(df.drop(df.columns[-1], axis=1), alpha=1-.9999994)
            datasets_summary.loc[i,"mvn_2"] = mvn[2]
            datasets_summary.loc[i, "mvn_5"] = mvn_5[2]
        print("Analysed dataset", dataset_id)

print(datasets_summary)
#datasets_summary.to_csv(os.path.join(root_dir,"datasets_summary.csv"))
# Save also in latex format. However, the table is split in two parts and concatenated, since it is too vertically big
ds1 = datasets_summary.iloc[:(len(datasets_summary)+1)//2].reset_index(drop=True)
ds2 = datasets_summary.iloc[(len(datasets_summary)+1)//2:].reset_index(drop=True)
ds_concat = pd.concat([ds1, ds2], axis=1)
#ds_concat.to_latex(os.path.join(root_dir, "datasets_summary.tex"), index=False)