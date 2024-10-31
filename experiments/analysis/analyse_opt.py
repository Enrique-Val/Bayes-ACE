import os
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import re

from experiments.utils import bh_test

# Path to dataset root
root_dir = "../results/exp_1_tmp/"

# Wilcoxon test alternative hypothesis
wx_alt = ["two-sided", "greater", "less"]

# Get all the values for penalty, dataset_id, models and n_vertex
def load_data(root_dir):
    file_pattern = re.compile(r"results_data(\d+)\.csv")
    results = {}
    # Get datasets ids
    for file in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file)
        if os.path.isdir(file_path):
            dataset_id = file_pattern.match(file).group(1)
            results[dataset_id] = pd.read_csv(file_path)
    return results


# Run the main function when the script is executed
if __name__ == "__main__":
    # Load the data
    data_dict = load_data(root_dir)

    # Best param dataset
    pd.DataFrame(columns=["eta_crossover", "eta_mutation", "selection_type"], index=list(data_dict.keys()))

    # Perform the BH test for every loaded dataset
    for dataset_id in data_dict.keys():
        data = data_dict[dataset_id]
        friedman_bh_results = bh_test(data)
        sp.critical_difference_diagram(friedman_bh_results["summary_ranks"], friedman_bh_results["p_adjusted"],
                                       label_fmt_left="{label} vertices", label_fmt_right="{label} vertices")
        plt.title(f"Data: {dataset_id}")
        plt.show()
        # Get the best column of data according to the BH test
        best_column = friedman_bh_results["summary_ranks"].idxmin()
        best_params = data.columns[best_column]
        # best_params is a str. Convert it back to dict
        best_params = dict([param.split("=") for param in best_params.split(",")])
        print(best_params)