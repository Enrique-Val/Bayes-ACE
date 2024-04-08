import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

# Check significant in n_vertex for each penalty
for penalty in penalties:
    print("Penalty: ", penalty)
    ver_list = np.array([])
    dist_list = np.array([])
    for dataset_id in dataset_ids:
        for network_type in ["CLG", "SP"]:
            data = datasets[(dataset_id, penalty, network_type)]
            for i in data.columns:
                ver_list = np.concatenate((ver_list, np.array([i] * len(data[i]))))
                dist_list = np.concatenate((dist_list, data[i]))
    to_analyse = pd.DataFrame(
        data={"n_vertex": ver_list.astype(float), "distance": dist_list})

    print("Initiating ANOVA analysis")
    model = ols('distance ~ C(n_vertex)', data=to_analyse).fit()
    print(sm.stats.anova_lm(model, typ=1))
    print(pairwise_tukeyhsd(endog=to_analyse['distance'], groups=to_analyse['n_vertex'], alpha=0.05))
