import os
import torch
import numpy as np
from tqdm import tqdm
import pyro.distributions as dist
import pyro.distributions.transforms as T


def save_model(model, optimizer, epoch, best_state, args):
    def f():
        if args.save:
            if args.verbose:
                print("Saving model..")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.load or args.path, "checkpoint.pt"),
            )
        best_state["model"] = model.state_dict()
        best_state["optimizer"] = optimizer.state_dict()
        best_state["epoch"] = epoch

    return f


def load_model(model, optimizer, best_state, args, load_start_epoch=False):
    def f():
        if args.save:
            if args.verbose:
                print("Loading model..")
            checkpoint = torch.load(os.path.join(args.load or args.path, "checkpoint.pt"))
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            if load_start_epoch:
                args.start_epoch = checkpoint["epoch"]
        elif best_state["model"] is not None:
            model.load_state_dict(best_state["model"])
            optimizer.load_state_dict(best_state["optimizer"])

    return f


def compute_log_p_x(model, x_mb):
    y_mb, log_diag_j_mb = model(x_mb)
    log_p_y_mb = (
        torch.distributions.Normal(torch.zeros_like(y_mb), torch.ones_like(y_mb))
        .log_prob(y_mb)
        .sum(-1)
    )
    return log_p_y_mb + log_diag_j_mb


def train_nf_model(model : dist.TransformedDistribution, optimizer, scheduler, data_loader_train, data_loader_valid, args):
    best_state = {"model": None, "optimizer": None, "epoch": 0}
    early_stopping_patience = args.early_stopping
    early_stopping_counter = 0

    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        total_loss = 0
        for batch_data in data_loader_train:
            optimizer.zero_grad()
            loss = -model.log_prob(batch_data[0]).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            model.clear_cache()

        if True:  # epoch % 500 == 0:
            print('step: {}, loss: {}'.format(epoch, total_loss / len(data_loader_train)))

        # Validation loop
        validation_loss = -torch.stack(
            [
                model.log_prob(batch_data[0]).mean()
                for batch_data in data_loader_train
            ],
            -1,
        ).mean()

        print(f'Epoch {epoch}, Validation Loss: {validation_loss}')

        # Update learning rate scheduler
        scheduler.step(validation_loss)

        # Early stopping check
        if validation_loss < best_loss:
            best_loss = validation_loss
            early_stopping_counter = 0
            best_model_params = model.base_dist.mean.detach().clone(), flow_dist.base_dist.stddev.detach().clone()
            for transform in model.transforms:
                best_model_params += tuple(param.detach().clone() for param in transform.parameters())

            print("Updated!")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(
                    "Validation loss hasn't improved for {} epochs. Early stopping...".format(early_stopping_patience))
                break

    # Restore best model parameters
    if best_model_params:
        model.base_dist.loc = best_model_params[0]
        model.base_dist.scale = best_model_params[1]
        current_param = 2
        for transform in model.transforms:
            for param in transform.parameters():
                param.data = best_model_params[current_param]
                current_param += 1

    validation_loss = -torch.stack(
        [
            model.log_prob(batch_data[0])
            for batch_data in data_loader_valid
        ],
        -1,
    ).mean()

    print(f'Final {1000}, Validation Loss: {validation_loss}')
    return epoch
