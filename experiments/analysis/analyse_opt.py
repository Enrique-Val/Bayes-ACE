import os
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import re

from experiments.utils import friedman_posthoc

# Path to dataset root
root_dir = "../results/exp_opt/"

# Wilcoxon test alternative hypothesis
wx_alt = ["two-sided", "greater", "less"]

# Get all the values for penalty, dataset_id, models and n_vertex
def load_data(root_dir):
    file_pattern = re.compile(r"results_data(\d+)\.csv")
    results = {}
    # Get datasets ids
    for file in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file)
        if not os.path.isdir(file_path):
            dataset_id = file_pattern.match(file).group(1)
            data = pd.read_csv(file_path, index_col=0)
            # Substitute nan with large number
            data = data.fillna(1e300)
            results[dataset_id] = data
    return results


# Run the main function when the script is executed
if __name__ == "__main__":
    # Load the data
    data_dict = load_data(root_dir)

    # Best param dataset
    pd.DataFrame(columns=["eta_crossover", "eta_mutation", "selection_type"], index=list(data_dict.keys()))

    # Perform the test for the combined dataset
    combined_data = pd.concat(data_dict.values()).reset_index(drop=True)
    avg_rank = combined_data.rank(axis=1, pct=True).mean()
    # Select best nine columns
    best_columns = avg_rank.nsmallest(9).index
    best_data = combined_data[best_columns]
    # Perform the Friedman test with BH post-hoc test
    test_results = friedman_posthoc(best_data, correct="bergmann")
    sp.critical_difference_diagram(test_results["summary_ranks"], test_results["p_adjusted"])
    plt.title(f"Data: Combined")
    plt.savefig(fname=root_dir+"plots/combined.png", bbox_inches="tight")
    plt.show(bbox_inches="tight")
    # Clear plt
    plt.clf()

    # Perform the BH test for every loaded dataset
    for dataset_id in data_dict.keys():
        data:pd.DataFrame = data_dict[dataset_id]
        # Compute the ranks
        avg_rank = data.rank(axis=1, pct=True).mean()
        print(avg_rank)

        # Select best nine columns
        best_columns = avg_rank.nsmallest(9).index
        best_data = data[best_columns]

        # Perform the Friedman test with BH post-hoc test
        test_results = friedman_posthoc(best_data, correct="bergmann")
        #plt.figure(figsize=(30, 10), dpi=100)
        sp.critical_difference_diagram(test_results["summary_ranks"], test_results["p_adjusted"])
        plt.title(f"Data: {dataset_id}")
        #sp.sign_plot(test_results["p_values"])
        plt.savefig(fname=root_dir+"plots/"+str(dataset_id)+".png", bbox_inches="tight")
        plt.show()
        #Clear plt
        plt.clf()
        '''# Get the best column of data according to the BH test
        best_column = friedman_bh_results["summary_ranks"].idxmin()
        best_params = data.columns[best_column]
        # best_params is a str. Convert it back to dict
        best_params = dict([param.split("=") for param in best_params.split(",")])
        print(best_params)'''