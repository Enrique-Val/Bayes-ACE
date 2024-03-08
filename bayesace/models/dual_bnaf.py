import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bayesace import get_and_process_data
# from models.BNAF.bnaf import BNAF
from models.BNAF.density_estimation import *


class Arguments():
    def __init__(self, dataset_id):
        self.device = "cuda:0"
        self.dataset_id = dataset_id #44091 44130 44123 44122 44127
        self.learning_rate = 1e-2
        self.batch_dim = 200
        self.clip_norm = 0.1
        self.epochs = 15  # 1000

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
        self.save = True
        self.tensorboard = "tensorboard"


class DualBNAF:
    def __init__(self, args):
        self.args = args
        self.true_model = self.create_model(args)
        self.false_model = self.create_model(args)

        self.true_frequency = 0.5  # Initial probability distribution (adjust as needed)
        self.false_frequency = 1 - self.true_frequency

    def load_dataset_oml(self):
        data = get_and_process_data(self.dataset_id)

        np.array_split(data, 3)

        dataset_train = torch.utils.data.TensorDataset(
            torch.from_numpy(dataset.trn.x).float().to(args.device)
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.args.batch_dim, shuffle=True
        )

        dataset_valid = torch.utils.data.TensorDataset(
            torch.from_numpy(dataset.val.x).float().to(args.device)
        )
        data_loader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=self.args.batch_dim, shuffle=False
        )

        dataset_test = torch.utils.data.TensorDataset(
            torch.from_numpy(dataset.tst.x).float().to(args.device)
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=self.args.batch_dim, shuffle=False
        )

        self.args.n_dims = dataset.n_dims

        return data_loader_train, data_loader_valid, data_loader_test

    def create_single_bnad(self, args):
        print("Loading dataset..")
        data_loader_train, data_loader_valid, data_loader_test = self.load_dataset(args)

        if args.save and not args.load:
            print("Creating directory experiment..")
            os.mkdir(args.path)
            with open(os.path.join(args.path, "args.json"), "w") as f:
                json.dump(args.__dict__, f, indent=4, sort_keys=True)

        print("Creating BNAF model..")
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

    def create_model(self, args):
        def create_single_bnaf():
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

        # Create BNAF models for both classes
        true_model = create_single_bnaf()
        false_model = create_single_bnaf()

        return true_model, false_model

    def train_single_bnaf(self, model, data_loader, optimizer, scheduler, epoch, args):
        model.train()
        for (x_mb,) in data_loader:
            loss = -compute_log_p_x(model, x_mb).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
            optimizer.step()

        validation_loss = -torch.stack([compute_log_p_x(model, x_mb).mean().detach() for x_mb, in data_loader],
                                       -1).mean()
        scheduler.step(validation_loss)

    def train_dual_bnaf(self, data_loader_true, data_loader_false, optimizer, scheduler, args):
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            # Train the model for the TRUE class
            self.train_single_bnaf(self.true_model, data_loader_true, optimizer, scheduler, epoch, args)

            # Train the model for the FALSE class
            self.train_single_bnaf(self.false_model, data_loader_false, optimizer, scheduler, epoch, args)

        # Update class probabilities based on the number of instances
        total_instances = len(data_loader_true.dataset) + len(data_loader_false.dataset)
        self.true_frequency = len(data_loader_true.dataset) / total_instances
        self.false_frequency = 1 - self.true_frequency

    def predict(self, x, threshold=0.5):
        true_log_prob = compute_log_p_x(self.true_model, x).mean().detach().item()
        false_log_prob = compute_log_p_x(self.false_model, x).mean().detach().item()

        prob_true_class = torch.sigmoid(true_log_prob - false_log_prob)
        return prob_true_class > threshold


# Usage of the DualBNAFClassifier class
def main():
    # Configuration and data loading
    args = argparse.Namespace(
        # Define your parameters here
    )

    dual_bnaf_classifier = DualBNAFClassifier(args)

    # Load datasets for True and False classes
    data_loader_true = load_true_data(args)
    data_loader_false = load_false_data(args)

    # Configure optimizer and learning rate scheduler
    optimizer = Adam(
        list(dual_bnaf_classifier.true_model.parameters()) + list(dual_bnaf_classifier.false_model.parameters()),
        lr=args.learning_rate, amsgrad=True, polyak=args.polyak)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.decay, patience=args.patience, cooldown=args.cooldown,
                                  min_lr=args.min_lr, verbose=True, early_stopping=args.early_stopping,
                                  threshold_mode="abs")

    dual_bnaf_classifier.train_dual_bnaf(data_loader_true, data_loader_false, optimizer, scheduler, args)

    # Example prediction
    input_data = torch.tensor(...)  # Input data for prediction
    is_true = dual_bnaf_classifier.predict(input_data)
    print(f"Prediction: {is_true}")


if __name__ == "__main__":
    main()
