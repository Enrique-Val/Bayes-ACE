import openml as oml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Convert np.e to float 64



x = np.e**np.array(-1e25, dtype=np.float64)
y = np.e**np.array(-1e25, dtype=np.float64)

def sum_log(x, y):
    return __sum_log__(np.log(x,), np.log(y))

def __sum_log__(logx, logy):
    return logx + np.log(1 + np.e ** (logy - logx))


if __name__ == "__main__" :
    print(np.e)
    print(np.log(x+y))
    #print(sum_log(x, y))
    print(__sum_log__(-1e25,-1e25))