import random

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from bayesace.models.BNAF_base.utils import compute_log_p_x, save_model, load_model, train_bnaf_model
from models.NF_base.utils import train_nf_model

torch.backends.cudnn.deterministic=True
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
import datetime

import json
import matplotlib.pyplot as plt

import pyro.distributions as dist
import pyro.distributions.transforms as T


class NfEstimator():
    def __init__(self, args):
        self.args = args

        base_dist = dist.Normal(torch.zeros(args.n_dims), torch.ones(args.n_dims))
        spline_transform = T.spline_coupling(args.n_dims)
        spline_transform_2 = T.spline_coupling(args.n_dims)
        self.model = dist.TransformedDistribution(base_dist, [spline_transform, spline_transform_2])
        self.optimizer = Adam(
            self.model.parameters(), lr=args.learning_rate, amsgrad=True, weight_decay=args.weight_decay
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            factor=args.decay,
            patience=args.patience,
            cooldown=args.cooldown,
            min_lr=args.min_lr,
            verbose=True,
            early_stopping=args.early_stopping,
            threshold_mode="abs",
        )

    def set_optimizer(self, optimizer):
        old_optimizer = self.optimizer
        self.optimizer = optimizer
        return old_optimizer

    def set_scheduler(self, scheduler) :
        old_scheduler = self.scheduler
        self.scheduler = scheduler
        return old_scheduler

    def compute_log_p_x(self, x_mb):
        if isinstance(x_mb, np.ndarray):
            x_mb = torch.from_numpy(x_mb).float().to(self.args.device)
        return compute_log_p_x(self.model, x_mb)

    def train(
            self,
            data_loader_train,
            data_loader_valid,
            data_loader_test =None,
            seed = 0
    ):
        torch.manual_seed(seed)
        random.seed(seed)
        self.args.start_epoch = 0

        ran_epochs = train_nf_model(
            self.model,
            self.optimizer,
            self.scheduler,
            data_loader_train,
            data_loader_valid,
            self.args,
        )

        validation_loss = -torch.stack(
            [self.compute_log_p_x(x_mb).mean().detach() for x_mb, in data_loader_valid],
            -1,
        ).mean()

        if data_loader_test is not None :
            test_loss = -torch.stack(
                [self.compute_log_p_x(x_mb).mean().detach() for x_mb, in data_loader_test], -1
            ).mean()

        if self.args.verbose :
            print("###### Stop training after {} epochs!".format(ran_epochs + 1))
            print("Validation loss: {:4.3f}".format(validation_loss.item()))
            if data_loader_test is not None:
                print("Test loss:       {:4.3f}".format(test_loss.item()))

        if self.args.save:
            with open(os.path.join(self.args.load or self.args.path, "results.txt"), "a") as f:
                print("###### Stop training after {} epochs!".format(ran_epochs + 1), file=f)
                print("Validation loss: {:4.3f}".format(validation_loss.item()), file=f)
                if data_loader_test is not None:
                    print("Test loss:       {:4.3f}".format(test_loss.item()), file=f)
