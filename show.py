import pandas as pd
import numpy as np

dataset_id = 44130 # 44091 44123 44127 44130 44122
penalty = 1
network_type = "CLG"

data = pd.read_csv("./results/exp_1/distances_data"+str(dataset_id)+"_net"+network_type+"_penalty"+str(penalty)+".csv")#, columns = ["1","5","10","15","20"])
data = data.drop(columns=["Unnamed: 0"],axis = 1)

data.columns = ["0", "1", "2", "3", "4", "5"]

print(data)
data = data.div(data.sum(axis=1), axis=0)
print(data)

print(data.mean(axis=0))

