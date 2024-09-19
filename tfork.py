import time

import numpy as np
from matplotlib import pyplot as plt

from bayesace.models.conditional_nvp import ConditionalNVP
from bayesace.models.utils import get_data, preprocess_data
import multiprocessing as mp


def cprint(model,p) :
    print(p)
    return p

# Define how the preprocessing will be done
JIT_COEF = 2
ELIM_OUTL = True
min_unique_vals = 50
max_unique_vals_to_jit = 0.05
max_cum_values = 3
minimum_spike_jitter = 0.0

if __name__ == "__main__":
    # Get data and plot every feature histogram
    data = get_data(44123, standardize=True)
    print(len(data))
    data_pr = preprocess_data(data, standardize=True, eliminate_outliers=ELIM_OUTL, jit_coef=JIT_COEF,
                                  min_unique_vals=min_unique_vals,
                                  max_unique_vals_to_jit=max_unique_vals_to_jit * len(data), max_instances=300000,
                                  minimum_spike_jitter=minimum_spike_jitter, max_cum_values=max_cum_values)

    for i in data.columns:
        if i != "class":
            try :
                print(i, "unique values: ", data[i].nunique())
                data_pr[i].hist(bins=500, alpha=0.5)
                data[i].hist(bins=500, alpha = 0.5)
                plt.title(i+" Unique: "+str(data[i].nunique()))
                plt.legend(["Preprocessed", "Original"])
                plt.show()
                time.sleep(0.1)
            except :
                pass

