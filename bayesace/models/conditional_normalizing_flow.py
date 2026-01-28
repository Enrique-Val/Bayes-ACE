import numpy as np
import pandas as pd
import torch
from sympy.physics.quantum.matrixutils import to_numpy

from bayesace.models.conditional_density_estimator import ConditionalDE


class NanLogProb(Exception):
    pass


class ConditionalNF(ConditionalDE):
    def __init__(self, gpu_acceleration=False, verbose = False):
        super().__init__()

        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu_acceleration else "cpu")
        self.trained = False
        self.verbose = verbose
        self.probs_tensor = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        super().fit(X, y)
        self.probs_tensor = torch.tensor(
            list(self.class_distribution.values()),
            device=self.device,
            dtype=torch.float32
        )

    def get_loaders(self, dataset, batch_size, proportion=0.8):
        dataset = dataset.copy()
        # Transform dataset to numpy and cast class from string to numerical
        class_column = np.zeros(len(dataset))
        for i, label in enumerate(self.class_distribution.keys()):
            class_column[dataset[self.class_var_name] == label] = i
        dataset[self.class_var_name] = class_column
        #dataset = dataset.astype(float)
        dataset_numpy = dataset.to_numpy()

        # Train validation split
        train_dataset, val_dataset = np.split(dataset_numpy,
                                              [int(proportion * len(dataset))])
        train_dataset_tensor = torch.utils.data.TensorDataset(
            torch.from_numpy(train_dataset).to(self.device, dtype=torch.get_default_dtype())
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset_tensor, batch_size=batch_size, shuffle=True, num_workers=0
        )

        val_dataset_tensor = torch.utils.data.TensorDataset(
            torch.from_numpy(val_dataset).to(self.device, dtype=torch.get_default_dtype())
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset_tensor, batch_size=batch_size, shuffle=False, num_workers=0
        )

        return train_loader, val_loader


    def logl_array(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # To be implemented by specific classes, depending on the implementation of the conditional distribution
        pass

    def logl_tensor(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # To be implemented by specific classes, depending on the implementation of the conditional distribution
        pass

    def logl(self, X: pd.DataFrame | np.ndarray | torch.Tensor, y : pd.DataFrame | np.ndarray | torch.Tensor=None,
             return_type = "tensor", torch_dtype = torch.float32) -> np.ndarray | torch.Tensor:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.device, dtype=torch_dtype)
        if y is not None:
            if isinstance(y, pd.Series):
                # Convert to numpy
                y = y.to_numpy()
            if isinstance(y, np.ndarray):
                # Convert to torch tensor. But first, convert from categorical strings to numerical for compatibility with torch
                class_labels = list(self.class_distribution.keys())
                class_column = np.zeros(y.shape[0], dtype=int)
                for i, label in enumerate(class_labels):
                    class_column[y == label] = i
                # Convert to torch tensor, use default dtype as defined
                y = torch.from_numpy(class_column).to(self.device, dtype=torch_dtype)

            if return_type == "numpy":
                return self.logl_tensor(X, y).detach().cpu().numpy()
            return self.logl_tensor(X,y)
        else:
            # --- Optimized Tensor Else Block ---
            log_likelihoods = []
            num_samples = X.shape[0]
            num_classes = len(self.class_distribution)

            # 1. Iterate over classes using Tensor operations
            for i in range(num_classes):
                # Create a tensor filled with the current class index 'i'
                # We use the same device/dtype as X to avoid transfer overhead
                y_class = torch.full((num_samples,), i, device=self.device, dtype=torch_dtype)

                # Compute log-likelihood using the tensor-optimized method
                log_likelihoods.append(self.logl_tensor(X, y_class))

            # 2. Stack results: Shape becomes (num_classes, num_samples)
            log_likelihoods_tensor = torch.stack(log_likelihoods, dim=0)

            # 3. Log-Sum-Exp (Marginalize over classes)
            # torch.logsumexp applies the max/exp/sum/log trick efficiently and stably
            lls = torch.logsumexp(log_likelihoods_tensor, dim=0)

            if return_type == "numpy":
                return lls.detach().cpu().numpy()
            return lls

    def predict_proba(self, X: pd.DataFrame | np.ndarray | torch.Tensor, output="tensor",
                      torch_dtype=torch.float32) -> np.ndarray | pd.DataFrame | torch.Tensor:
        # 1. Standardize Input to Tensor
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.device, dtype=torch_dtype)
        elif isinstance(X, torch.Tensor):
            X = X.to(self.device, dtype=torch_dtype)

        # 2. Compute Log-Likelihoods (Logits)
        # We calculate log P(x, Y) for every class Y
        log_likelihoods = []
        num_samples = X.shape[0]
        num_classes = len(self.class_distribution)
        class_keys = list(self.class_distribution.keys())

        for i in range(num_classes):
            # Create a target tensor filled with class index 'i'
            y_class = torch.full((num_samples,), i, device=self.device, dtype=torch_dtype)
            log_likelihoods.append(self.logl_tensor(X, y_class))

        # Stack to shape: (num_classes, num_samples)
        logits = torch.stack(log_likelihoods, dim=0)

        # 3. Compute Probabilities via Softmax
        # P(Y|x) = exp(log P(x,Y)) / sum(exp(log P(x,Y)))
        # torch.softmax is numerically stable (handles overflow/underflow automatically)
        probs = torch.softmax(logits, dim=0)

        # Transpose to shape (num_samples, num_classes)
        probs = probs.transpose(0, 1)

        # 4. Handle Edge Cases (If all logits are -inf, softmax returns NaN)
        # This replicates the original logic: zero_l = np.where(p_x == 0) -> uniform dist
        if torch.isnan(probs).any():
            nan_mask = torch.isnan(probs).any(dim=1)
            probs[nan_mask] = 1.0 / num_classes

        # 5. Output Formatting
        if output == "tensor":
            return probs

        # Move to CPU for Numpy/Pandas
        probs_np = probs.detach().cpu().numpy()

        if output == "pandas":
            return pd.DataFrame(probs_np, columns=class_keys)

        return probs_np
