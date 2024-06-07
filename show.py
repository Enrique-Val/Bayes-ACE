import time

import pandas as pd
import numpy as np
import openml as oml
import matplotlib.pyplot as plt

from bayesace import get_data, hill_climbing

data = get_data(44130)

data_old = data.copy()

#print(data_old["prevFixDur"])
#raise Exception()

for i in data.columns[:-1]:
    data = data[data[i] < data[i].std() * 3]
    data = data[data[i] > -data[i].std() * 3]
    data[i] = data[i] + np.random.normal(0, 0.3 / (len(data) ** (1 / 5)), data[i].shape)

network = hill_climbing(data_old, "CLG")
gaussian_sample = network.sample(len(data), ordered=True)

#print(data["pH"])
#raise Exception("Fin")

print(data.dtypes)

n_bins = 150
print(len(data_old))
for j in data.columns[:-1]:
    time.sleep(0.2)
    plt.hist(data_old[j], bins=n_bins, density=False, color="purple")
    plt.hist(data[j], bins=n_bins, density=False, alpha=0.6, color="skyblue")
    #plt.hist(gaussian_sample[j], bins=n_bins, density=True, alpha=0.4, color="red")
    plt.title(str(j))
    plt.show()


'''
for i in [44091, 44122,44123,44127,44130]
    data = oml.datasets.get_dataset(i, download_data=True, download_qualities=False,
                                    download_features_meta_data=False).get_data()[0]'''