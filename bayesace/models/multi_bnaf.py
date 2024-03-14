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


class Arguments():
    def __init__(self, dataset_id):
        self.device = "cpu"#"cuda:0"
        self.dataset_id = dataset_id #44091 44130 44123 44122 44127
        self.learning_rate = 1e-2
        self.batch_dim = 20
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
    #full_data = get_and_process_data(args.dataset_id)
    print(os.getcwd())
    full_data = pd.read_csv("../../toy-3class.csv")
    full_data["class"] = full_data["z"].astype('category')
    full_data = full_data.drop("z", axis=1)
    feature_columns = [i for i in full_data.columns if i != "class"]
    full_data[feature_columns] = StandardScaler().fit_transform(full_data[feature_columns].values)

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

def create_model(args, verbose=False):

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
    params = sum(
        (p != 0).sum() if len(p.shape) > 1 else torch.tensor(p.shape).item()
        for p in model.parameters()
    ).item()


    return model


def save_model(model, optimizer, epoch, args):
    def f():
        if args.save:
            print("Saving model..")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.load or args.path, "checkpoint.pt"),
            )

    return f


def load_model(model, optimizer, args, load_start_epoch=False):
    def f():
        if args.save :
            print("Loading model..")
            checkpoint = torch.load(os.path.join(args.load or args.path, "checkpoint.pt"))
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            if load_start_epoch:
                args.start_epoch = checkpoint["epoch"]

    return f


def compute_log_p_x(model, x_mb):
    y_mb, log_diag_j_mb = model(x_mb)
    log_p_y_mb = (
        torch.distributions.Normal(torch.zeros_like(y_mb), torch.ones_like(y_mb))
        .log_prob(y_mb)
        .sum(-1)
    )
    return log_p_y_mb + log_diag_j_mb


def train(
    model,
    optimizer,
    scheduler,
    data_loader_train,
    data_loader_valid,
    data_loader_test,
    args,
):

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(os.path.join(args.tensorboard, args.load or args.path))

    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        t = tqdm(data_loader_train, smoothing=0, ncols=80)
        train_loss = []

        for (x_mb,) in t:
            loss = -compute_log_p_x(model, x_mb).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            t.set_postfix(loss="{:.2f}".format(loss.item()), refresh=False)
            train_loss.append(loss)

        train_loss = torch.stack(train_loss).mean()
        optimizer.swap()
        validation_loss = -torch.stack(
            [
                compute_log_p_x(model, x_mb).mean().detach()
                for x_mb, in data_loader_valid
            ],
            -1,
        ).mean()
        optimizer.swap()

        print(
            "Epoch {:3}/{:3} -- train_loss: {:4.3f} -- validation_loss: {:4.3f}".format(
                epoch + 1,
                args.start_epoch + args.epochs,
                train_loss.item(),
                validation_loss.item(),
            )
        )
        my_copy = model
        stop = scheduler.step(
            validation_loss,
            callback_best=save_model(model, optimizer, epoch + 1, args),
            callback_reduce=load_model(model, optimizer, args),
        )

        if args.tensorboard:
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
            writer.add_scalar("loss/validation", validation_loss.item(), epoch + 1)
            writer.add_scalar("loss/train", train_loss.item(), epoch + 1)

        if stop:
            break
    #load_model(model, optimizer, args)()
    optimizer.swap()
    validation_loss = -torch.stack(
        [compute_log_p_x(model, x_mb).mean().detach() for x_mb, in data_loader_valid],
        -1,
    ).mean()
    test_loss = -torch.stack(
        [compute_log_p_x(model, x_mb).mean().detach() for x_mb, in data_loader_test], -1
    ).mean()

    print("###### Stop training after {} epochs!".format(epoch + 1))
    print("Validation loss: {:4.3f}".format(validation_loss.item()))
    print("Test loss:       {:4.3f}".format(test_loss.item()))

    if args.save:
        with open(os.path.join(args.load or args.path, "results.txt"), "a") as f:
            print("###### Stop training after {} epochs!".format(epoch + 1), file=f)
            print("Validation loss: {:4.3f}".format(validation_loss.item()), file=f)
            print("Test loss:       {:4.3f}".format(test_loss.item()), file=f)

def create_single_bnaf(args, i, data_loader_train, data_loader_valid, data_loader_test):

    args.path = os.path.join(
        "checkpoint_"+str(i),
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
        os.mkdir("checkpoint_"+str(i))
        os.mkdir(args.path)
        with open(os.path.join(args.path, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4, sort_keys=True)

    print("Creating BNAF_base model..")
    model = create_model(args, verbose=True)

    print("Creating optimizer..")
    optimizer = Adam(
        model.parameters(), lr=args.learning_rate, amsgrad=True, polyak=args.polyak
    )

    print("Creating scheduler..")
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=args.decay,
        patience=args.patience,
        cooldown=args.cooldown,
        min_lr=args.min_lr,
        verbose=True,
        early_stopping=args.early_stopping,
        threshold_mode="abs",
    )

    args.start_epoch = 0
    if args.load:
        load_model(model, optimizer, args, load_start_epoch=True)()

    print("Training..")
    train(
        model,
        optimizer,
        scheduler,
        data_loader_train,
        data_loader_valid,
        data_loader_test,
        args,
    )
    return model

class MultiBNAF:
    def __init__(self, args):
        self.args = args
        class_data_loaders, self.class_dist = load_dataset_oml(self.args)
        self.bnafs = {}
        for i in class_data_loaders.keys():
            data_loader_train, data_loader_valid, data_loader_test = class_data_loaders[i]
            model = create_single_bnaf(self.args, i, data_loader_train, data_loader_valid, data_loader_test)
            self.bnafs[i] = model


if __name__ == "__main__":
    args = Arguments(44130)
    args.epochs = 2000
    multi_bnaf = MultiBNAF(args)
    print(multi_bnaf.class_dist)
    #print(multi_bnaf.bnafs)
    print(multi_bnaf.bnafs["a"](torch.from_numpy(np.array([[0,0]])).float()))
    print("High",compute_log_p_x(multi_bnaf.bnafs["a"],torch.from_numpy(np.array([[-1,0]])).float()))
    print("Low", compute_log_p_x(multi_bnaf.bnafs["a"],torch.from_numpy(np.array([[2,2]])).float()))
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
                torch.exp(compute_log_p_x(model, x_mb)).detach()
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