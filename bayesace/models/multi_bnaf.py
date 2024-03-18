import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from bayesace import get_and_process_data
# from models.BNAF_base.bnaf import BNAF
import os
import datetime

from bayesace.models.BNAF_base.bnaf import MaskedWeight, Tanh, BNAF, Permutation, Sequential
from bayesace.models.BNAF_base.optim.adam import Adam
from bayesace.models.BNAF_base.optim.lr_scheduler import ReduceLROnPlateau
import json
import matplotlib.pyplot as plt

from BNAF_base.bnaf_estimator import BnafEstimator


class Arguments():
    def __init__(self, dataset_id):
        self.device = "cpu" #"cuda:0"
        self.dataset_id = dataset_id #44091 44130 44123 44122 44127
        self.learning_rate = 1e-2
        self.batch_dim = 200
        self.clip_norm = 0.1
        self.epochs = 100  # 1000

        self.patience = 20
        self.cooldown = 10
        self.early_stopping = 100
        self.decay = 0.5
        self.min_lr = 5e-4
        self.polyak = 0.998

        self.flows = 5
        self.layers = 1
        self.hidden_dim = 10
        self.residual = "gated"  # choices=[None, "normal", "gated"]

        self.expname = ""
        self.load = None
        self.save = True#True
        self.tensorboard = "tensorboard"

def load_dataset_oml(args):
    full_data = get_and_process_data(args.dataset_id)
    print(os.getcwd())
    '''full_data = pd.read_csv("../../toy-3class.csv")
    full_data["class"] = full_data["z"].astype('category')
    full_data = full_data.drop("z", axis=1)
    feature_columns = [i for i in full_data.columns if i != "class"]
    full_data[feature_columns] = StandardScaler().fit_transform(full_data[feature_columns].values)'''

    class_data_loaders = {}
    class_dist = full_data["class"].value_counts(normalize = True).to_dict()
    for i in np.unique(full_data["class"]):
        dataset_class = full_data[full_data["class"] == i].drop(columns=["class"]).values

        d_train, d_validate, d_test = np.split(dataset_class,
                                               [int(.6 * len(dataset_class)), int(.8 * len(dataset_class))])

        dataset_train = torch.utils.data.TensorDataset(
            torch.from_numpy(d_train).float().to(args.device)
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_dim, shuffle=True
        )

        dataset_valid = torch.utils.data.TensorDataset(
            torch.from_numpy(d_validate).float().to(args.device)
        )
        data_loader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=args.batch_dim, shuffle=False
        )

        dataset_test = torch.utils.data.TensorDataset(
            torch.from_numpy(d_test).float().to(args.device)
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.batch_dim, shuffle=False
        )

        class_data_loaders[i]= (data_loader_train, data_loader_valid, data_loader_test)

    args.n_dims = len(full_data.columns) - 1

    return class_data_loaders, class_dist

def create_single_bnaf(args, i, data_loader_train, data_loader_valid, data_loader_test):
    dir_data = "checkpoint_data"+str(args.dataset_id)
    dir_class = dir_data+"/"+dir_data+"_class_"+str(i)
    args.path = os.path.join(
        dir_class,
        "{}{}_layers{}_h{}_flows{}{}_{}".format(
            args.expname + ("_" if args.expname != "" else ""),
            args.dataset_id,
            args.layers,
            args.hidden_dim,
            args.flows,
            "_" + args.residual if args.residual else "",
            str(datetime.datetime.now())[:-7].replace(" ", "-").replace(":", "-"),
        ),
    )

    if args.save and not args.load:
        print("Creating directory experiment..")
        if not os.path.exists(dir_data):
            os.mkdir(dir_data)
        if not os.path.exists(dir_class):
            os.mkdir(dir_class)
        os.mkdir(args.path)
        with open(os.path.join(args.path, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4, sort_keys=True)

    print("Creating BNAF_base model..")
    bnaf = BnafEstimator(args)

    print("Training..")
    bnaf.train(
        data_loader_train,
        data_loader_valid,
        data_loader_test
    )
    return bnaf

class MultiBnaf:
    def __init__(self, args):
        self.args = args
        class_data_loaders, self.class_dist = load_dataset_oml(self.args)
        self.bnafs = {}
        for i in class_data_loaders.keys():
            data_loader_train, data_loader_valid, data_loader_test = class_data_loaders[i]
            model = create_single_bnaf(self.args, i, data_loader_train, data_loader_valid, data_loader_test)
            self.bnafs[i] = model


if __name__ == "__main__":
    torch.manual_seed(0)
    args = Arguments(44091)
    args.epochs = 2000
    multi_bnaf = MultiBnaf(args)
    print(multi_bnaf.class_dist)
    #print(multi_bnaf.bnafs)
    print(multi_bnaf.bnafs["a"].model(torch.from_numpy(np.array([[0,0]])).float()))
    print("High",multi_bnaf.bnafs["a"].compute_log_p_x(torch.from_numpy(np.array([[-1,0]])).float()))
    print("Low", multi_bnaf.bnafs["a"].compute_log_p_x(torch.from_numpy(np.array([[2,2]])).float()))
    limit = 3
    step = 0.01
    for i in ["a","b","c"] :
        model = multi_bnaf.bnafs[i]
        grid = torch.Tensor(
            [
                [a, b]
                for a in np.arange(-limit, limit, step)
                for b in np.arange(-limit, limit, step)
            ]
        )
        grid_dataset = torch.utils.data.TensorDataset(grid.to(args.device))
        grid_data_loader = torch.utils.data.DataLoader(
            grid_dataset, batch_size=1000, shuffle=False
        )

        prob = torch.cat(
            [
                torch.exp(model.compute_log_p_x(x_mb)).detach()
                for x_mb, in grid_data_loader
            ],
            0,
        )

        prob = prob.view(int(2 * limit / step), int(2 * limit / step)).t()

        if False:
            prob = prob.clamp(max=prob.mean() + 3 * prob.std())

        plt.figure(figsize=(12, 12))
        plt.imshow(prob.cpu().data.numpy(), extent=(-limit, limit, -limit, limit))
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.show()