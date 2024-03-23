import numpy as np
import pandas as pd
import torch

from bayesace.models.utils import hill_climbing

torch.backends.cudnn.deterministic = True

import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# from bayesace.utils import get_and_process_data, hill_climbing
# from models.BNAF_base.bnaf import BNAF
import os
import datetime

from bayesace.models.BNAF_base.bnaf import MaskedWeight, Tanh, BNAF, Permutation, Sequential
from bayesace.models.BNAF_base.optim.adam import Adam
from bayesace.models.BNAF_base.optim.lr_scheduler import ReduceLROnPlateau
import json
import matplotlib.pyplot as plt
import multiprocessing as mp

from bayesace.models.BNAF_base.bnaf_estimator import BnafEstimator
import pybnesian as pb


class Arguments():
    def __init__(self):
        self.device = "cuda"  # "cuda:0"
        self.dataset_id = 1
        self.learning_rate = 1e-2
        self.batch_dim = 200
        self.clip_norm = 0.1
        self.epochs = 2000  # 1000

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
        self.save = True  # True
        self.tensorboard = "tensorboard"
        self.test_data = True
        self.verbose = False


def get_data_loaders(args, data):
    full_data = data
    class_data_loaders = {}
    class_dist = full_data["class"].value_counts(normalize=True).to_dict()
    for i in np.unique(full_data["class"]):
        dataset_class = full_data[full_data["class"] == i].drop(columns=["class"]).values

        d_list = None
        if args.test_data:
            d_list = np.split(dataset_class,
                              [int(.6 * len(dataset_class)), int(.8 * len(dataset_class))])
        else:
            d_list = np.split(dataset_class,
                              [int(.8 * len(dataset_class))])
        data_loaders = []
        for j, d in enumerate(d_list):
            dataset_tensor = torch.utils.data.TensorDataset(
                torch.from_numpy(d).float().to(args.device)
            )
            flag = False
            if j == 0:
                flag = True
            data_loader = torch.utils.data.DataLoader(
                dataset_tensor, batch_size=args.batch_dim, shuffle=flag, num_workers=0
            )
            data_loaders.append(data_loader)

        # If there is no test, append a None value to mark it
        if len(data_loaders) == 2:
            data_loaders.append(None)

        class_data_loaders[i] = tuple(data_loaders)

    args.n_dims = len(full_data.columns) - 1

    return class_data_loaders, class_dist


def create_single_bnaf(args, i, data_loaders, seed=0,
                       verbose=False) -> BnafEstimator:
    data_loader_train, data_loader_valid, data_loader_test = data_loaders
    dir_data = "checkpoint_data" + str(args.dataset_id)
    dir_class = dir_data + "/" + dir_data + "_class_" + str(i)
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
        if verbose:
            print("Creating directory experiment..")
        if not os.path.exists(dir_data):
            os.mkdir(dir_data)
        if not os.path.exists(dir_class):
            os.mkdir(dir_class)
        os.mkdir(args.path)
        with open(os.path.join(args.path, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4, sort_keys=True)

    if verbose:
        print("Creating BNAF model..")
    bnaf = BnafEstimator(args)

    if verbose:
        print("Training..")
    bnaf.train(
        data_loader_train,
        data_loader_valid,
        data_loader_test,
        seed=seed
    )
    return bnaf


class MultiBnaf:
    def __init__(self, args, data, seed=0, parallelize=False):
        self.args = args
        class_data_loaders, self.class_dist = get_data_loaders(self.args, data)
        self.bnafs = {}
        if parallelize:
            pool = mp.Pool(len(class_data_loaders.keys()))
            result = pool.starmap(create_single_bnaf, [(args, i, class_data_loaders[i], seed, args.verbose) for i in
                                                       class_data_loaders.keys()])
            pool.close()
            for i, label in enumerate(class_data_loaders.keys()):
                self.bnafs[label] = result[i]
        else:
            for label in class_data_loaders.keys():
                self.bnafs[label] = create_single_bnaf(args, label, class_data_loaders[label], seed, args.verbose)
        self.sampler = hill_climbing(data=data, bn_type="CLG")

    def get_class_labels(self):
        return list(self.class_dist.keys()).copy()

    def get_class_distribution(self):
        return self.class_dist.copy()

    def sample(self, n_samples, ordered=True, seed=None):
        return self.sampler.sample(n_samples, ordered=ordered, seed=seed)

    def logl(self, data, class_var_name="class"):
        to_ret = np.zeros(data.shape[0])
        for i in self.class_dist.keys():
            data_i = data[data[class_var_name] == i].drop(class_var_name, axis=1)
            index_i = data_i.index
            if len(index_i) == 0:
                continue
            logl_i = self.bnafs[i].compute_log_p_x(data_i.values).detach().cpu().numpy()
            to_ret[index_i] = logl_i + np.log(self.class_dist[i])
        return to_ret

    def likelihood(self, data, class_var_name="class"):
        # If the class variable is passed, remove it
        if class_var_name in data.columns:
            data = data.drop(class_var_name, axis=1)
        data = data.values
        to_ret = np.zeros(data.shape[0])
        for i in self.class_dist.keys():
            logl_i = self.bnafs[i].compute_log_p_x(data).detach().cpu().numpy()
            to_ret = to_ret + np.e ** (logl_i + np.log(self.class_dist[i]))
        return to_ret

    def predict(self, data: np.ndarray):
        ll = np.zeros(data.shape[0])
        acc = np.zeros((len(self.class_dist.keys()), data.shape[0]))
        for i, label in enumerate(self.class_dist.keys()):
            logl_i = self.bnafs[label].compute_log_p_x(data).detach().cpu().numpy() + np.log(self.class_dist[label])
            ll = ll + np.e ** logl_i
            acc[i] = np.e ** logl_i
        return acc.transpose() / ll[:, None]


'''
if __name__ == "__main__":
    torch.manual_seed(0)
    args = Arguments()
    args.epochs = 5
    args.verbose = True
    args.test_data = True
    data = get_and_process_data(44091)
    multi_bnaf = MultiBnaf(args, data)
    print(multi_bnaf.class_dist)
    # print(multi_bnaf.bnafs)
    print(multi_bnaf.bnafs["a"].model(torch.from_numpy(np.array([[0, 0]])).float()))
    print("High", multi_bnaf.bnafs["a"].compute_log_p_x(torch.from_numpy(np.array([[-1, 0]])).float()))
    print("Low", multi_bnaf.bnafs["a"].compute_log_p_x(torch.from_numpy(np.array([[2, 2]])).float()))
    limit = 3
    step = 0.01
    for i in ["a", "b", "c"]:
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
        plt.show()'''
