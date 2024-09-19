import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np

edge_len = 5
hyp = ((edge_len/2)**2+edge_len)**0.5

vertices = [
    (0, 0),
    (edge_len, 0),
    (edge_len*1.5, hyp),
    (edge_len, hyp*2),
    (0, hyp*2),
    (-edge_len*0.5, hyp)
]


unzipped = list(zip(*vertices))

n = 20
flag = True
data = pd.DataFrame([], columns=["x", "y", "class"])
for i in vertices :
    sample = multivariate_normal(mean=i, cov=[[0.2, 0], [0, 0.2]]).rvs(n)
    data_i = pd.DataFrame(sample, columns=["x", "y"])
    data_i["class"] = flag
    data = data._append(data_i)
    flag = not flag

data = data.sample(frac=1).reset_index(drop=True)
data.to_csv("six_gaussians_small.csv", index=False)
plt.scatter(data["x"], data["y"], c=data["class"])
plt.show()