import os
import torch
import numpy as np
from tqdm import tqdm


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


def train_bnaf_model(model, optimizer, scheduler, data_loader_train, data_loader_valid, args):
    best_state = {"model": None, "optimizer": None, "epoch": 0}

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(os.path.join(args.tensorboard, args.load or args.path))

    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        t = data_loader_train
        if args.verbose:
            t = tqdm(data_loader_train, smoothing=0, ncols=80)
        train_loss = []

        for (x_mb,) in t:
            loss = -compute_log_p_x(model, x_mb).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            if args.verbose:
                t.set_postfix(loss="{:.2f}".format(loss.item()), refresh=False)
            train_loss.append(loss)

        train_loss = torch.stack(train_loss).mean()
        optimizer.swap()
        validation_loss = -torch.stack(
            [
                compute_log_p_x(model,x_mb).mean().detach()
                for x_mb, in data_loader_valid
            ],
            -1,
        ).mean()
        optimizer.swap()
        print("Validation loss: ",validation_loss.item())

        if args.verbose:
            print(
                "Epoch {:3}/{:3} -- train_loss: {:4.3f} -- validation_loss: {:4.3f}".format(
                    epoch + 1,
                    args.start_epoch + args.epochs,
                    train_loss.item(),
                    validation_loss.item(),
                )
            )
        stop = scheduler.step(
            validation_loss,
            callback_best=save_model(model, optimizer, epoch + 1, best_state, args),
            callback_reduce=load_model(model, optimizer, best_state, args),
        )

        if args.tensorboard:
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
            writer.add_scalar("loss/validation", validation_loss.item(), epoch + 1)
            writer.add_scalar("loss/train", train_loss.item(), epoch + 1)

        if stop:
            break
    load_model(model, optimizer, best_state, args)()
    return epoch
