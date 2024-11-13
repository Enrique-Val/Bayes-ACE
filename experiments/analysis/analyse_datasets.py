import os
import pandas as pd
from pingouin import multivariate_normality

# Iterate over all datasets in the folder
root_dir = "../results/exp_cv_2/"

datasets_summary = pd.DataFrame(columns=["dataset_id", "n attributes", "n instances", "mvn_5"])
for i,dataset_id in enumerate(os.listdir(root_dir)):
    dataset_path = os.path.join(root_dir, dataset_id)
    if os.path.isdir(dataset_path):
        file = os.path.join(dataset_path, "resampled_data" + dataset_id + ".csv")
        df = pd.read_csv(file, index_col=0)
        datasets_summary.loc[i] = [dataset_id, df.shape[1] - 1, df.shape[0], True]
        df = df.head(1000)
        mvn = multivariate_normality(df.drop(columns=["class"]), alpha=0.05)
        mvn_5 = multivariate_normality(df.drop(columns=["class"]), alpha=1-.9999994)
        #datasets_summary.loc[i,"mvn_3"] = mvn[2]
        datasets_summary.loc[i, "mvn_5"] = mvn_5[2]

print(datasets_summary)
datasets_summary.to_csv("../results/exp_cv_2/datasets_summary.csv")
# Save also in latex format. However, the table is split in two parts and concatenated, since it is too vertically big
ds1 = datasets_summary.iloc[:(len(datasets_summary)+1)//2].reset_index(drop=True)
ds2 = datasets_summary.iloc[(len(datasets_summary)+1)//2:].reset_index(drop=True)
ds_concat = pd.concat([ds1, ds2], axis=1)
ds_concat.to_latex(os.path.join(root_dir, "datasets_summary.tex"), index=False)


'''
# Check if the dataset is MVN using the pingouin test for all class values
    for class_value in df["class"].unique():
        df_class = df[df["class"] == class_value]
        mvn = multivariate_normality(df_class.drop(columns=["class"]), alpha=0.05)
        mvn_5 = multivariate_normality(df_class.drop(columns=["class"]), alpha=1-.9999994)
    datasets_summary.loc[i,"mvn_3"] = mvn[2] and datasets_summary.loc[i,"mvn_3"]
    datasets_summary.loc[i, "mvn_5"] = mvn_5[2] and datasets_summary.loc[i, "mvn_5"]
'''