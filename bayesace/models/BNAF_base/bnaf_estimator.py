import random

import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.deterministic=True
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
import datetime

from bayesace.models.BNAF_base.bnaf import MaskedWeight, Tanh, BNAF, Permutation, Sequential
from bayesace.models.BNAF_base.optim.adam import Adam
from bayesace.models.BNAF_base.optim.lr_scheduler import ReduceLROnPlateau
import json
import matplotlib.pyplot as plt


def create_model(args):
    flows = []
    for f in range(args.flows):
        layers = []
        for _ in range(args.layers - 1):
            layers.append(
                MaskedWeight(
                    args.n_dims * args.hidden_dim,
                    args.n_dims * args.hidden_dim,
                    dim=args.n_dims,
                )
            )
            layers.append(Tanh())

        flows.append(
            BNAF(
                *(
                        [
                            MaskedWeight(
                                args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims
                            ),
                            Tanh(),
                        ]
                        + layers
                        + [
                            MaskedWeight(
                                args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims
                            )
                        ]
                ),
                res=args.residual if f < args.flows - 1 else None
            )
        )

        if f < args.flows - 1:
            flows.append(Permutation(args.n_dims, "flip"))

    model = Sequential(*flows).to(args.device)
    return model



class BnafEstimator():
    def __init__(self, args):
        self.args = args

        self.model = create_model(self.args)
        self.optimizer = Adam(
            self.model.parameters(), lr=args.learning_rate, amsgrad=True, polyak=args.polyak
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

        self.best_state = {"model" : None, "optimizer" : None, "epoch" : 0}

    def set_optimizer(self, optimizer):
        old_optimizer = self.optimizer
        self.optimizer = optimizer
        return old_optimizer

    def set_scheduler(self, scheduler) :
        old_scheduler = self.scheduler
        self.scheduler = scheduler
        return old_scheduler

    def save_model(self, epoch):
        def f():
            if self.args.save:
                if self.args.verbose :
                    print("Saving model..")
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(self.args.load or self.args.path, "checkpoint.pt"),
                )
            self.best_state["model"] = self.model.state_dict()
            self.best_state["optimizer"] = self.optimizer.state_dict()
            self.best_state["epoch"] = epoch

        return f

    def load_model(self, load_start_epoch=False):
        def f():
            if self.args.save:
                if self.args.verbose :
                    print("Loading model..")
                checkpoint = torch.load(os.path.join(self.args.load or self.args.path, "checkpoint.pt"))
                self.model.load_state_dict(checkpoint["model"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])

                if load_start_epoch:
                    self.args.start_epoch = checkpoint["epoch"]
            elif self.best_state["model"] is not None :
                self.model.load_state_dict(self.best_state["model"])
                self.optimizer.load_state_dict(self.best_state["optimizer"])

        return f

    def compute_log_p_x(self, x_mb):
        if isinstance(x_mb, np.ndarray) :
            x_mb = torch.from_numpy(x_mb).float().to(self.args.device)
        y_mb, log_diag_j_mb = self.model(x_mb)
        log_p_y_mb = (
            torch.distributions.Normal(torch.zeros_like(y_mb), torch.ones_like(y_mb))
            .log_prob(y_mb)
            .sum(-1)
        )
        return log_p_y_mb + log_diag_j_mb

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

        if self.args.tensorboard :
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(os.path.join(self.args.tensorboard, self.args.load or self.args.path))

        epoch = self.args.start_epoch
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.epochs):

            t = data_loader_train
            if self.args.verbose :
                t = tqdm(data_loader_train, smoothing=0, ncols=80)
            train_loss = []

            for (x_mb,) in t:
                loss = -self.compute_log_p_x(x_mb).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.args.verbose :
                    t.set_postfix(loss="{:.2f}".format(loss.item()), refresh=False)
                train_loss.append(loss)

            train_loss = torch.stack(train_loss).mean()
            self.optimizer.swap()
            validation_loss = -torch.stack(
                [
                    self.compute_log_p_x(x_mb).mean().detach()
                    for x_mb, in data_loader_valid
                ],
                -1,
            ).mean()
            self.optimizer.swap()

            if self.args.verbose :
                print(
                    "Epoch {:3}/{:3} -- train_loss: {:4.3f} -- validation_loss: {:4.3f}".format(
                        epoch + 1,
                        self.args.start_epoch + self.args.epochs,
                        train_loss.item(),
                        validation_loss.item(),
                    )
                )
            stop = self.scheduler.step(
                validation_loss,
                callback_best=self.save_model(epoch + 1),
                callback_reduce=self.load_model(),
            )

            if self.args.tensorboard:
                writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch + 1)
                writer.add_scalar("loss/validation", validation_loss.item(), epoch + 1)
                writer.add_scalar("loss/train", train_loss.item(), epoch + 1)

            if stop:
                break
        self.load_model()()
        self.optimizer.swap()
        validation_loss = -torch.stack(
            [self.compute_log_p_x(x_mb).mean().detach() for x_mb, in data_loader_valid],
            -1,
        ).mean()

        if data_loader_test is not None :
            test_loss = -torch.stack(
                [self.compute_log_p_x(x_mb).mean().detach() for x_mb, in data_loader_test], -1
            ).mean()

        if self.args.verbose :
            print("###### Stop training after {} epochs!".format(epoch + 1))
            print("Validation loss: {:4.3f}".format(validation_loss.item()))
            if data_loader_test is not None:
                print("Test loss:       {:4.3f}".format(test_loss.item()))

        if self.args.save:
            with open(os.path.join(self.args.load or self.args.path, "results.txt"), "a") as f:
                print("###### Stop training after {} epochs!".format(epoch + 1), file=f)
                print("Validation loss: {:4.3f}".format(validation_loss.item()), file=f)
                if data_loader_test is not None:
                    print("Test loss:       {:4.3f}".format(test_loss.item()), file=f)
