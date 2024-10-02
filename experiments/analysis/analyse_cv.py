import os

import pandas as pd

from scipy.stats import wilcoxon

# Path to dataset root
root_dir = "../results/exp_cv_2/"

metrics_test = {"Logl_mean": "less", "Brier_mean": "greater", "AUC_mean": "less", "Time_mean": "less"}

rd_dfs = {}
sd_dfs = {}
for i in metrics_test.keys():
    rd_dfs[i] = pd.DataFrame()
    sd_dfs[i] = pd.DataFrame()

for dataset_id in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, dataset_id)
    if os.path.isdir(dataset_path):
        file = os.path.join(dataset_path, "data_" + dataset_id + ".csv")
        df = pd.read_csv(file, index_col=0)
        for i in rd_dfs.keys():
            rd_dfs[i] = pd.concat((rd_dfs[i], df.loc[[i], ["CLG_RD", "GT_RD"]]))
        try:
            for i in rd_dfs.keys():
                sd_dfs[i] = pd.concat((sd_dfs[i], df.loc[[i], ["CLG", "NF"]]))
        except:
            continue

for i in rd_dfs.keys():
    rd_dfs[i] = rd_dfs[i].astype(float)
    sd_dfs[i] = sd_dfs[i].astype(float)

for i in metrics_test.keys():
    print("Testing if", i, "is", metrics_test[i], "for Bayesian networks vs Normalising flows")
    wxr = wilcoxon(rd_dfs[i]["CLG_RD"], rd_dfs[i]["GT_RD"], alternative=metrics_test[i])
    wxs = wilcoxon(sd_dfs[i]["CLG"], sd_dfs[i]["NF"], alternative=metrics_test[i])
    print("Real data:", wxr)
    print("Generated data:", wxs)
    print()
