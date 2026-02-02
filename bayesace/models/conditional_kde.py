import torch
import numpy as np
import pandas as pd
from typing import Union

# Assuming ConditionalDE is in a file named bayesace
from bayesace import ConditionalDE


class ConditionalKDE(ConditionalDE):
    def __init__(self, bandwidth=1.0, device="cpu"):
        """
        PyTorch-based Conditional KDE.
        bandwidth (h) corresponds to the standard deviation (sigma) of the Gaussian Kernel.
        """
        super().__init__()
        self.bandwidth = bandwidth
        self.device = device
        # We store training data as tensors per class
        self.X_train_: dict[any, torch.Tensor] = {}
        self.log_priors_: dict[any, float] = {}

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        super().fit(X, y) 

        X_np = X.to_numpy()
        y_np = y.to_numpy() if isinstance(y, pd.Series) else y

        # Convert class priors to log priors for numerical stability
        for cls in self.get_class_labels():
            # Store data as tensor on device
            X_cls = X_np[y_np == cls]
            self.X_train_[cls] = torch.tensor(X_cls, dtype=torch.float32, device=self.device)
            self.log_priors_[cls] = np.log(self.class_distribution[cls])

        self.trained = True

    def _kde_log_prob(self, X: torch.Tensor, class_label) -> torch.Tensor:
        """
        Computes log P(X | Class) using PyTorch broadcasting.
        Formula: LogSumExp( -0.5 * ||x - xi||^2 / h^2 ) - constants
        """
        train_samples = self.X_train_[class_label]  # Shape: (N_train, D)

        # Calculate squared Euclidean distance: ||x - xi||^2
        # Using torch.cdist (computes L2 norm), then squaring it
        # X shape: (Batch, D), train_samples: (N_train, D)
        # dists shape: (Batch, N_train)
        dists = torch.cdist(X, train_samples, p=2)
        sq_dists = dists ** 2

        # Gaussian Kernel Log-Likelihood
        # log( exp( -sq_dist / (2h^2) ) ) -> -sq_dist / (2h^2)
        variance = self.bandwidth ** 2
        log_kernels = -sq_dists / (2 * variance)

        # LogSumExp trick to sum probabilities in log-space
        # log P(x) = log(1/N * sum(exp(kernels)))
        #          = logsumexp(kernels) - log(N)
        log_sum = torch.logsumexp(log_kernels, dim=1)

        # Add normalization constants: - (D/2) * log(2*pi) - D * log(h)
        n_features = X.shape[1]
        n_samples_train = train_samples.shape[0]

        const = -0.5 * n_features * np.log(2 * np.pi) - n_features * np.log(self.bandwidth)
        log_prob = log_sum - np.log(n_samples_train) + const

        return log_prob

    def logl(self, X: Union[pd.DataFrame, np.ndarray, torch.Tensor], y=None, return_type="tensor"):
        """
        Returns Log-Likelihood. Differentiable if X is a Tensor.
        """
        # 1. Handle Input Type (Preserve Gradients if Tensor)
        input_is_tensor = isinstance(X, torch.Tensor)
        if input_is_tensor:
            X_torch = X.float().to(self.device)
        elif isinstance(X, np.ndarray):
            X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            # Convert pandas/numpy to tensor
            X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
            X_torch = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        batch_size = X_torch.shape[0]

        # 2. Compute Joint Log-Likelihood log P(X, y) if y is provided
        if y is not None:
            # Handle y format
            if isinstance(y, (pd.Series, np.ndarray)):
                y_np = y.to_numpy() if isinstance(y, pd.Series) else y
            else:
                # If y is a tensor, detach to numpy for indexing (labels are discrete)
                y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

            log_likelihood = torch.full((batch_size,), -float('inf'), device=self.device)

            present_labels = np.unique(y_np)
            for label in present_labels:
                if label in self.X_train_:
                    mask = (y_np == label)
                    # Convert mask to indices for torch
                    indices = np.where(mask)[0]
                    if len(indices) == 0: continue

                    # P(X, y) = P(X | y) * P(y)
                    # log P(X, y) = log P(X | y) + log P(y)
                    log_p_x_given_y = self._kde_log_prob(X_torch[indices], label)
                    log_likelihood[indices] = log_p_x_given_y + self.log_priors_[label]

            return log_likelihood if return_type=="tensor" else log_likelihood

        # 3. Compute Marginal Log-Likelihood log P(X) if y is None
        else:
            # log P(X) = log sum_c exp( log P(X|c) + log P(c) )
            class_log_probs = []

            for label in self.get_class_labels():
                log_p_x_given_y = self._kde_log_prob(X_torch, label)
                log_joint = log_p_x_given_y + self.log_priors_[label]
                class_log_probs.append(log_joint)

            # Stack: (n_classes, n_samples)
            class_log_probs = torch.stack(class_log_probs, dim=0)

            # LogSumExp over classes (dim 0)
            marginal_logl = torch.logsumexp(class_log_probs, dim=0)

            return marginal_logl if input_is_tensor else marginal_logl.detach().cpu().numpy()

    def predict_proba(self, X, output="tensor"):
        """
        Overridden to allow gradient flow.
        """
        is_tensor = isinstance(X, torch.Tensor)
        # Calculate log P(X) (Marginal) -> Shape (N,)
        log_marginal = self.logl(X, y=None)

        if not is_tensor:
            # Re-wrap as tensor to continue calculation in torch
            X_torch = torch.tensor(X if not isinstance(X, pd.DataFrame) else X.values,
                                   dtype=torch.float32, device=self.device)
            log_marginal_torch = torch.tensor(log_marginal, device=self.device)
        else:
            X_torch = X
            log_marginal_torch = log_marginal

        # Calculate P(Y|X) = exp( log P(X|Y) + log P(Y) - log P(X) )
        probs_list = []
        for cls in self.get_class_labels():
            log_joint = self._kde_log_prob(X_torch, cls) + self.log_priors_[cls]
            # log P(Y|X)
            log_posterior = log_joint - log_marginal_torch
            probs_list.append(torch.exp(log_posterior))

        probs = torch.stack(probs_list, dim=1)  # Shape (N, n_classes)

        if output == "tensor":
            return probs
        elif output == "pandas":
            return pd.DataFrame(probs.detach().cpu().numpy(), columns=self.get_class_labels())
        elif output == "numpy" and not is_tensor:
            return probs.detach().cpu().numpy()
        return probs  # Return tensor

    def sample(self, n_samples: int, seed=None):
        # 1. Set global seeds for both Torch (sampling) and Numpy (shuffling)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        samples = []
        samples_class_label = []
        classes = self.get_class_labels()

        for cls in classes:
            # Calculate how many samples we need for this class
            n_samples_cls = int(n_samples * self.class_distribution[cls]) + 1

            # 2. Call the pure-torch sampler
            # We pass seed=None to ensure we don't reset the RNG every iteration
            sample_i = self.sample_given_class(n_samples_cls, cls, seed=None)

            samples.append(sample_i)
            samples_class_label.append(np.repeat(cls, n_samples_cls))

        # 3. Concatenate and format results
        samples = np.concatenate(samples, axis=0)
        samples_class_label = np.concatenate(samples_class_label, axis=0)

        # Trim excess samples
        samples_class_label = samples_class_label[:n_samples]
        samples = pd.DataFrame(samples, columns=self.columns).head(n_samples)

        samples[self.class_var_name] = samples_class_label
        samples[self.class_var_name] = pd.Categorical(samples[self.class_var_name], categories=classes)

        # Shuffle (uses numpy RNG, seeded at start)
        samples = samples.sample(frac=1, random_state=seed)

        return samples

    def sample_given_class(self, n_samples: int, class_label, seed=None):
        """
        Pure PyTorch sampling from KDE.
        Logic:
        1. Randomly select N training points (with replacement).
        2. Add Gaussian noise scaled by bandwidth.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if class_label not in self.X_train_:
            raise ValueError(f"Class {class_label} not found in training data.")

        # Get stored training tensor for this class
        # Shape: (N_train, D)
        train_samples = self.X_train_[class_label]
        n_train, n_features = train_samples.shape

        # --- Step 1: Select Centers ---
        # Generate random indices to pick which training points act as centers
        # We pick 'n_samples' indices from range [0, n_train)
        indices = torch.randint(low=0, high=n_train, size=(n_samples,), device=self.device)

        # Gather the centers: Shape (n_samples, D)
        selected_centers = train_samples[indices]

        # --- Step 2: Add Gaussian Noise ---
        # Generate standard normal noise N(0, 1) and scale by bandwidth
        noise = torch.randn(size=(n_samples, n_features), device=self.device) * self.bandwidth

        # --- Step 3: Combine ---
        # x_new = x_center + noise
        samples = selected_centers + noise

        # Return as numpy to be compatible with the parent 'sample' method
        return samples.cpu().numpy()


