import os
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Path to dataset root
root_dir = "../results/exp_1/"

# Regex to match filenames like distances_data44123_pen1.csv
file_pattern = re.compile(r"distances_data(\d+)_penalty(\d+)\.csv")


# Function to load and organize data from the directory
def load_data(root_dir):
    data_by_penalty = {}

    for dataset_id in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_id)

        if os.path.isdir(dataset_path):
            for model in os.listdir(dataset_path):  # "clg" and "nf" subfolders (models)
                model_path = os.path.join(dataset_path, model)

                if os.path.isdir(model_path):
                    for file in os.listdir(model_path):
                        match = file_pattern.match(file)
                        if match:
                            # Extract dataset_id and penalty from filename
                            extracted_dataset_id = match.group(1)
                            penalty = match.group(2)

                            # Read CSV
                            file_path = os.path.join(model_path, file)
                            df = pd.read_csv(file_path, index_col=0)

                            # Initialize dictionary entry for this penalty if not exists
                            if penalty not in data_by_penalty:
                                data_by_penalty[penalty] = {}

                            # Add data to the respective dataset, model, and penalty
                            data_by_penalty[penalty][(extracted_dataset_id, model)] = df
    return data_by_penalty


# Function to run Friedman test and Nemenyi posthoc for each dataset/model/penalty
def perform_friedman_and_nemenyi(data_dict):
    results = {}

    for penalty, datasets_models in data_dict.items():
        print(datasets_models)
        results[penalty] = {}
        for (dataset_id, model), df in datasets_models.items():
            # Perform Friedman test across the vertex columns
            friedman_result = friedmanchisquare(*[df[vertex] for vertex in df.columns])

            # Check if Friedman test is significant
            if friedman_result.pvalue < 0.05:
                # Perform Nemenyi posthoc test
                nemenyi_result = sp.posthoc_nemenyi_friedman(df.values)
                results[penalty][(dataset_id, model)] = {
                    "friedman": friedman_result,
                    "nemenyi": nemenyi_result
                }
            else:
                results[penalty][(dataset_id, model)] = {
                    "friedman": friedman_result,
                    "nemenyi": None
                }

    return results


# Function to concatenate datasets for global analysis
def concatenate_datasets(data_by_penalty):
    concatenated_results = {}
    for penalty, datasets_models in data_by_penalty.items():
        concatenated_df = pd.concat([df for df in datasets_models.values()])
        concatenated_results[penalty] = concatenated_df
    return concatenated_results


# Function to plot Nemenyi test results as a heatmap
def plot_nemenyi_heatmap(nemenyi_result, dataset_id, model, penalty):
    plt.figure(figsize=(10, 8))
    sns.heatmap(nemenyi_result, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"Nemenyi Posthoc Test - Dataset: {dataset_id}, Model: {model}, Penalty: {penalty}")
    plt.xlabel("Vertex")
    plt.ylabel("Vertex")
    plt.show()


# Function to print and save the Friedman/Nemenyi results
def print_and_save_results(results, output_dir="results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for penalty, models_data in results.items():
        print(f"Penalty {penalty}:")

        for (dataset_id, model), test_results in models_data.items():
            friedman_p = test_results["friedman"].pvalue
            print(f"  Dataset: {dataset_id}, Model: {model}")
            print(f"    Friedman p-value: {friedman_p}")

            if test_results["nemenyi"] is not None:
                nemenyi_df = test_results["nemenyi"]
                print(f"    Nemenyi Test (Vertex Comparison):")
                print(nemenyi_df)

                # Save the Nemenyi results as a CSV
                nemenyi_file = f"nemenyi_data{dataset_id}_model{model}_pen{penalty}.csv"
                nemenyi_df.to_csv(os.path.join(output_dir, nemenyi_file))


def perform_wilcoxon_test(data_by_penalty):
    wilcoxon_results = {}

    for penalty, datasets_models in data_by_penalty.items():
        wilcoxon_results[penalty] = {}
        for dataset_id in {k[0] for k in datasets_models.keys()}:  # Unique dataset IDs
            # Ensure both models (clg and nf) are available for comparison
            if (dataset_id, "clg") in datasets_models and (dataset_id, "nf") in datasets_models:
                print(datasets_models)
                clg_df = datasets_models[(dataset_id, "clg")]
                nf_df = datasets_models[(dataset_id, "nf")]

                # Wilcoxon test across the values of all vertex columns (paired test)
                wilcoxon_stats = []
                for vertex in clg_df.columns:
                    stat, p_value = wilcoxon(clg_df[vertex], nf_df[vertex])
                    wilcoxon_stats.append((vertex, stat, p_value))

                wilcoxon_results[penalty][dataset_id] = wilcoxon_stats

    return wilcoxon_results

def perform_global_wilcoxon_test(data_by_penalty):
    global_wilcoxon_results = {}

    for penalty, datasets_models in data_by_penalty.items():
        # Collect and concatenate data across all datasets for clg and nf
        combined_clg = pd.concat([df for (dataset_id, model), df in datasets_models.items() if model == "clg"])
        combined_nf = pd.concat([df for (dataset_id, model), df in datasets_models.items() if model == "nf"])

        # Wilcoxon test across all vertex columns
        global_wilcoxon_stats = []
        for vertex in combined_clg.columns:
            stat, p_value = wilcoxon(combined_clg[vertex], combined_nf[vertex])
            global_wilcoxon_stats.append((vertex, stat, p_value))

        global_wilcoxon_results[penalty] = global_wilcoxon_stats

    return global_wilcoxon_results


# Function to print Wilcoxon test results
def print_wilcoxon_results(wilcoxon_results, output_dir="wilcoxon_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for penalty, datasets_data in wilcoxon_results.items():
        print(f"Penalty {penalty} - Wilcoxon Test:")
        for dataset_id, stats in datasets_data.items():
            print(f"  Dataset: {dataset_id}")
            for vertex, stat, p_value in stats:
                print(f"    Vertex {vertex}: stat = {stat}, p-value = {p_value}")

            # Save the Wilcoxon results to a CSV
            wilcoxon_file = f"wilcoxon_data{dataset_id}_pen{penalty}.csv"
            pd.DataFrame(stats, columns=["Vertex", "Statistic", "P-Value"]).to_csv(
                os.path.join(output_dir, wilcoxon_file), index=False
            )


# Run the main function when the script is executed
if __name__ == "__main__":
    # Load the data
    data_by_penalty = load_data(root_dir)

    print(data_by_penalty['1'][('44127', 'clg')].head())
    print(data_by_penalty['1'][('44127', 'nf')].head())
    raise Exception("Stop here")

    # Perform Friedman test and Nemenyi posthoc for each dataset/model/penalty
    friedman_nemenyi_results = perform_friedman_and_nemenyi(data_by_penalty)

    # Plot Nemenyi test results for each significant combination
    for penalty, models_data in friedman_nemenyi_results.items():
        print(models_data)
        for (dataset_id, model), test_results in models_data.items():
            if test_results["nemenyi"] is not None:
                plot_nemenyi_heatmap(test_results["nemenyi"], dataset_id, model, penalty)

    # Print and save the results
    #print_and_save_results(friedman_nemenyi_results)

    # For all datasets combined, concatenate and perform global analysis
    #combined_data_by_penalty = concatenate_datasets(data_by_penalty)
    #friedman_nemenyi_combined = perform_friedman_and_nemenyi({"combined": combined_data_by_penalty})

    # Print and save combined results
    print("Global Analysis (All Datasets Combined):")
    #print_and_save_results(friedman_nemenyi_combined, output_dir="global_results")

    # Perform Wilcoxon test between clg and nf for each dataset and penalty
    wilcoxon_results = perform_wilcoxon_test(data_by_penalty)
    print_wilcoxon_results(wilcoxon_results)

    # For all datasets combined, concatenate and perform global analysis
    combined_data_by_penalty = concatenate_datasets(data_by_penalty)
    friedman_nemenyi_combined = perform_friedman_and_nemenyi({"combined": combined_data_by_penalty})

    # Print and save combined results
    print("Global Analysis (All Datasets Combined):")
    print_and_save_results(friedman_nemenyi_combined, output_dir="global_results")

    # Perform Wilcoxon test globally across all datasets combined
    global_wilcoxon_results = perform_global_wilcoxon_test(data_by_penalty)
    print_wilcoxon_results({"combined": global_wilcoxon_results}, output_dir="global_wilcoxon_results")

