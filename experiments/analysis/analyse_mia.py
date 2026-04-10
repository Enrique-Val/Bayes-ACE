import os

import numpy as np
import pandas as pd

results_dir = "../results/mia"


# Iterate over the dataset ids in the results directory
for model in ["clg", "nf"]:
    aucs = []
    accs = []
    for dataset_id in os.listdir(results_dir)[:-1]:
        dataset_path = os.path.join(results_dir, dataset_id)
        if os.path.isdir(dataset_path):
            res = pd.read_csv(os.path.join(dataset_path, "mia_results_"+model+".csv"), index_col=0)
            aucs.append(res["Scenario_2_Signals_And_Features"].loc["AUC"])
            accs.append(res["Scenario_2_Signals_And_Features"].loc["Accuracy"])
    print("Model:", model)
    print("Average AUC:", sum(aucs)/len(aucs) ,"+/-", np.std(aucs))
    print("Average Accuracy:", sum(accs)/len(accs), "+/-", np.std(accs))
    print()
