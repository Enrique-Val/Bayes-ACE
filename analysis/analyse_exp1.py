import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import wilcoxon

folder = "../results/exp_1/"
dataset_ids = [44091, 44122, 44123, 44127, 44130]
penalties = [1, 5, 10, 15, 20]

datasets = {}
for dataset_id in dataset_ids:
    for penalty in penalties:
        for network_type in ["CLG", "SP"]:
            file_name = 'distances_data' + str(dataset_id) + '_net' + network_type + '_penalty' + str(penalty) + '.csv'
            results = pd.read_csv(folder + file_name, index_col=0)
            data = results.values
            data -= data.min(axis=1)[:, None]
            data /= data.ptp(axis=1)[:, None]
            results[:] = data
            datasets[(dataset_id, penalty, network_type)] = results

# Check if n_vertex increase with penalty
pen_list = np.array([])
ver_list = np.array([])
dist_list = np.array([])
for penalty in penalties:
    for dataset_id in dataset_ids:
        for network_type in ["CLG", "SP"]:
            data = datasets[(dataset_id, penalty, network_type)]
            for i in data.columns:
                pen_list = np.concatenate((pen_list, np.array([penalty] * len(data[i]))))
                ver_list = np.concatenate((ver_list, np.array([i] * len(data[i]))))
                dist_list = np.concatenate((dist_list, data[i]))
to_analyse = pd.DataFrame(
    data={"penalty": pen_list.astype(float), "n_vertex": ver_list.astype(float), "distance": dist_list})
print(to_analyse)

print("Initiating ANOVA analysis")
model = ols('distance ~ C(penalty) + C(n_vertex) + C(penalty):C(n_vertex)', data=to_analyse).fit()
print(sm.stats.anova_lm(model, typ=2))
print(pairwise_tukeyhsd(endog=to_analyse['distance'], groups=to_analyse['n_vertex'], alpha=0.05))

# Check, for every penalty, if distance becomes lesser
for penalty in penalties:
    penalty_df = to_analyse[to_analyse['penalty'] == penalty]
    print("Penalty:",penalty)
    for i in range(0,5):
        for j in range(i+1,6) :
            data_i = penalty_df[penalty_df['n_vertex'] == i]
            data_j = penalty_df[penalty_df['n_vertex'] == j]
            print("Compare",i,"with ",j)
            print(wilcoxon(data_i["distance"], data_j["distance"], alternative="greater"))
    print()

# Check if the number of variables and vertices is important to the number of evaluations

