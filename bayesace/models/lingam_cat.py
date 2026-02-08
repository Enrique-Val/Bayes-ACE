import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as dist
import lingam
from scipy.stats import t, norm
from sklearn.mixture import GaussianMixture
from bayesace import ConditionalDE


class LingamClassifier(ConditionalDE, nn.Module):
    def __init__(self, bin_edges=None, bin_names=None, random_state=42, prior_knowledge=None, device="cpu"):
        """
        Args:
            bin_edges (list of float): The cut-off points to discretize the continuous target.
                                       Do not include -inf or inf (they are added automatically).
                                       Example: [0.0] creates classes (-inf, 0] and (0, inf).
            bin_names (list of str):   Optional names for the resulting bins.
                                       Must have length = len(bin_edges) + 1.
                                       Example: ["Low", "High"] for bin_edges=[0.0].
        """
        ConditionalDE.__init__(self)  # Initialize ABC
        nn.Module.__init__(self)  # Initialize Torch Module

        self.device = device
        self.random_state = random_state
        self.prior_knowledge = prior_knowledge
        self.bin_names = bin_names

        # User provided discretization (Ordinal cut-points)
        if bin_edges is None:
            # Default fallback: split at 0 if nothing provided
            self.user_bin_edges = [0.0]
        else:
            self.user_bin_edges = sorted(list(bin_edges))

        # Parameters to be learned
        self.lingam_model = None
        self.noise_config_ = {}
        self.candidates = ['Normal', 'Laplace', 'StudentT', '2GMM', '3GMM', 'Logistic']
        self.class_labels_ = None  # Will store the final label array

        # Torch Parameters
        self.register_buffer('bin_edges', None)
        self.register_buffer('B', None)  # SEM Matrix for Features
        self.register_buffer('intercepts', None)  # Intercepts for Features
        self.register_buffer('target_coeffs', None)  # Coeffs for Target
        self.register_buffer('target_intercept', None)  # Intercept for Target

        self.trained = False

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        """
        Fits the LiNGAM model and estimates noise distributions.
        Note: 'y' must be the CONTINUOUS target variable.
        """
        super().fit(X, y)
        self.trained = False

        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # 1. Prepare Data for LiNGAM (X + Continuous Y)
        full_df = X.copy()
        target_col_temp = "target_variable_internal"
        full_df[target_col_temp] = y

        print(f"--- Fitting Structure (LiNGAM) ---")
        self.lingam_model = lingam.DirectLiNGAM(
            random_state=self.random_state,
            prior_knowledge=self.prior_knowledge
        )
        self.lingam_model.fit(full_df)

        # 2. Extract Matrices
        B_full = self.lingam_model.adjacency_matrix_

        # Calculate residuals: E = Data - (Data @ B.T + c)
        term_c_plus_e = full_df - np.dot(full_df, B_full.T)
        c_full = term_c_plus_e.mean(axis=0).values
        residuals = term_c_plus_e - c_full

        # 3. Fit Noise Distributions
        self._fit_noise_dists(residuals)

        # 4. Process User-Provided Discretization
        print(f"--- Configuring User-Provided Discretization ---")

        # Prepare edges: Add -inf and +inf to ensure full coverage
        edges_np = np.array(self.user_bin_edges, dtype=np.float32)

        # Prepend -inf if not present
        if not np.isneginf(edges_np[0]):
            edges_np = np.concatenate(([-np.inf], edges_np))

        # Append +inf if not present
        if not np.isposinf(edges_np[-1]):
            edges_np = np.concatenate((edges_np, [np.inf]))

        print(f"Final Bin Edges: {edges_np}")

        # Validate bin names
        num_bins = len(edges_np) - 1
        if self.bin_names is not None:
            if len(self.bin_names) != num_bins:
                raise ValueError(
                    f"Discretization created {num_bins} bins, but {len(self.bin_names)} names were provided. "
                    f"Edges: {self.user_bin_edges} -> {num_bins} regions.")
            self.class_labels_ = np.array(self.bin_names)
        else:
            self.class_labels_ = np.arange(num_bins)

        # 5. Convert to Torch Tensors and Store
        B_tensor = torch.tensor(B_full, dtype=torch.float32)
        c_tensor = torch.tensor(c_full, dtype=torch.float32)

        # SEM for Features
        self.B = B_tensor[:-1, :-1]
        self.intercepts = c_tensor[:-1]

        # Regression for Target
        self.target_coefficients = B_tensor[-1, :-1]
        self.target_intercept = c_tensor[-1]

        self.bin_edges = torch.tensor(edges_np, dtype=torch.float32)

        # Update class distribution based on the provided bins
        y_binned = pd.cut(y, bins=edges_np, labels=False, include_lowest=True)

        # Calculate distribution
        self.class_distribution = {}
        unique_classes = np.arange(num_bins)
        actual_classes, counts = np.unique(y_binned, return_counts=True)
        total_count = len(y)

        for c in unique_classes:
            if c in actual_classes:
                idx = np.where(actual_classes == c)[0][0]
                label = self.class_labels_[c]  # Use name if available
                self.class_distribution[label] = counts[idx] / total_count
            else:
                label = self.class_labels_[c]
                self.class_distribution[label] = 0.0

        self.to(self.device)
        self.trained = True

    def logl(self, X: pd.DataFrame | torch.Tensor,
             y: pd.Series | np.ndarray | torch.Tensor = None) -> np.ndarray | torch.Tensor:
        """
        Computes Log Likelihood.
        If y is provided, it must be the CLASS INDICES (0, 1, ...), not the string names,
        unless you implement a lookup. For standard optimization/evaluation, assume indices.
        """
        is_torch_input = isinstance(X, torch.Tensor)
        x_tensor, y_tensor = self._prepare_input(X, y)

        # 1. Compute Continuous Predictions (SEM)
        # x_hat = Parent(X) * Coeffs + Intercept
        x_hat = x_tensor @ self.B.T + self.intercepts
        x_residuals = x_tensor - x_hat

        # 2. Sum log_prob of residuals for features
        total_log_prob = torch.zeros(x_tensor.shape[0], device=self.device)

        for i in range(x_tensor.shape[1]):
            if i in self.noise_config_:
                dist_type, params = self.noise_config_[i]
                d = self._get_dist(dist_type, params)
                total_log_prob += d.log_prob(x_residuals[:, i])

        if y_tensor is None:
            return total_log_prob if is_torch_input else total_log_prob.detach().cpu().numpy()

        # 3. If y provided, add log P(y|X)
        y_continuous_hat = (x_tensor @ self.target_coefficients) + self.target_intercept

        # Get probabilities matrix [Batch, n_classes]
        probs = self._compute_proba_matrix(y_continuous_hat)

        # Gather specific probabilities for the provided Y labels
        # Note: y_tensor must contain integer indices here
        y_idx = y_tensor.long().view(-1, 1)
        selected_probs = probs.gather(1, y_idx).squeeze(1)

        # Avoid log(0)
        selected_probs = torch.clamp(selected_probs, min=1e-10)

        total_log_prob += torch.log(selected_probs)

        return total_log_prob if is_torch_input else total_log_prob.detach().cpu().numpy()

    def predict_proba(self, X: np.ndarray | torch.Tensor, output="numpy") -> np.ndarray | pd.DataFrame | torch.Tensor:
        is_torch_input = isinstance(X, torch.Tensor)
        x_tensor, _ = self._prepare_input(X)

        y_continuous_hat = (x_tensor @ self.target_coefficients) + self.target_intercept
        probs = self._compute_proba_matrix(y_continuous_hat)

        if output == "pandas":
            # Use the stored class labels (names or ints)
            return pd.DataFrame(probs.detach().cpu().numpy(), columns=self.get_class_labels())
        elif output == "numpy" and not is_torch_input:
            return probs.detach().cpu().numpy()
        return probs

    def predict(self, X: np.ndarray | torch.Tensor):
        """
        Predicts the class label (name if provided, else index).
        """
        probs = self.predict_proba(X, output="numpy")
        indices = np.argmax(probs, axis=1)
        return self.class_labels_[indices]

    def _compute_proba_matrix(self, y_continuous_hat):
        target_idx = len(self.B)
        dist_type, params = self.noise_config_[target_idx]
        latent_dist = self._get_dist(dist_type, params)

        n_classes = self.bin_edges.shape[0] - 1
        batch_size = y_continuous_hat.shape[0]
        probs = torch.zeros((batch_size, n_classes), dtype=torch.float32, device=self.device)

        for k in range(n_classes):
            lower_bound = self.bin_edges[k]
            upper_bound = self.bin_edges[k + 1]
            z_lower = lower_bound - y_continuous_hat
            z_upper = upper_bound - y_continuous_hat
            probs[:, k] = latent_dist.cdf(z_upper) - latent_dist.cdf(z_lower)

        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
        return probs

    def sample(self, n_samples: int, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        with torch.no_grad():
            n_features = self.B.shape[0]

            # 1. Sample Noise for Features
            noise_x = torch.zeros((n_samples, n_features), device=self.device)
            for i in range(n_features):
                if i in self.noise_config_:
                    dist_type, params = self.noise_config_[i]
                    d = self._get_dist(dist_type, params)
                    noise_x[:, i] = d.sample((n_samples,)).view(-1)

            # 2. Solve for Features X
            identity = torch.eye(n_features, device=self.device)
            adj_inv = torch.linalg.inv(identity - self.B)
            term = noise_x + self.intercepts.unsqueeze(0)
            x_generated = term @ adj_inv.T

            # 3. Sample Target Variable (Continuous)
            target_idx = n_features
            if target_idx in self.noise_config_:
                dist_type, params = self.noise_config_[target_idx]
                d = self._get_dist(dist_type, params)
                noise_y = d.sample((n_samples,)).view(-1)
            else:
                noise_y = torch.zeros(n_samples, device=self.device)

            y_cont = (x_generated @ self.target_coefficients) + self.target_intercept + noise_y

            # 4. Discretize
            y_indices = torch.bucketize(y_cont, self.bin_edges)
            y_indices = y_indices - 1
            n_classes = len(self.class_labels_)
            y_indices = torch.clamp(y_indices, 0, n_classes - 1)

            # 5. Format Output
            x_np = x_generated.cpu().numpy()
            y_idx_np = y_indices.cpu().numpy()

            # Map indices to names if available
            y_final = self.class_labels_[y_idx_np]

            cols = self.columns if self.columns is not None else [f"X{i}" for i in range(n_features)]
            df = pd.DataFrame(x_np, columns=cols)
            class_name = self.class_var_name if self.class_var_name else "class"
            df[class_name] = y_final

            return df

    # --- Internals ---
    def _prepare_input(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            x_tensor = torch.tensor(X.values, dtype=torch.float32, device=self.device)
        else:
            x_tensor = X.to(self.device) if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32,
                                                                                          device=self.device)
        y_tensor = None
        if y is not None:
            if isinstance(y, (pd.Series, np.ndarray, list)):
                y_tensor = torch.tensor(np.array(y), dtype=torch.float32, device=self.device)
            else:
                y_tensor = y.to(self.device)
        return x_tensor, y_tensor

    def _fit_noise_dists(self, residuals_df):
        n_samples = len(residuals_df)
        tensor_res = torch.tensor(residuals_df.values, dtype=torch.float32)
        for i, col in enumerate(residuals_df.columns):
            data = tensor_res[:, i]
            best_bic = float('inf')
            best_cfg = None
            for dtype in self.candidates:
                ll, params, k_params = self._fit_single_dist(data, dtype)
                bic = k_params * np.log(n_samples) - 2 * ll
                if bic < best_bic:
                    best_bic = bic
                    best_cfg = (dtype, params)
            self.noise_config_[i] = best_cfg

    def _fit_single_dist(self, data, dist_type):
        if dist_type == 'Normal':
            scale = torch.sqrt((data ** 2).mean())
            ll = dist.Normal(0.0, scale).log_prob(data).sum().item()
            return ll, {'scale': scale}, 1
        elif dist_type == 'Laplace':
            scale = torch.abs(data).mean()
            ll = dist.Laplace(0.0, scale).log_prob(data).sum().item()
            return ll, {'scale': scale}, 1
        elif dist_type == 'StudentT':
            try:
                params = t.fit(data.numpy(), floc=0)
                df, scale = params[0], params[2]
                ll = dist.StudentT(df=df, loc=0.0, scale=scale).log_prob(data).sum().item()
                return ll, {'df': torch.tensor(df), 'scale': torch.tensor(scale)}, 2
            except:
                return -np.inf, {}, 0
        elif 'GMM' in dist_type:
            try:
                k = int(dist_type[0])
                gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=self.random_state)
                gmm.fit(data.numpy().reshape(-1, 1))
                weights = torch.tensor(gmm.weights_, dtype=torch.float32)
                means = torch.tensor(gmm.means_.flatten(), dtype=torch.float32)
                scales = torch.sqrt(torch.tensor(gmm.covariances_.flatten(), dtype=torch.float32))
                mix = dist.Categorical(probs=weights)
                comp = dist.Normal(loc=means, scale=scales)
                ll = dist.MixtureSameFamily(mix, comp).log_prob(data).sum().item()
                return ll, {'weights': weights, 'means': means, 'scales': scales}, (3 * k - 1)
            except:
                return -np.inf, {}, 0
        elif "Logistic":
            from scipy.stats import logistic as sc_log
            loc, scale = sc_log.fit(data.numpy(), floc=0)
            base = dist.Uniform(0, 1)
            transforms = [dist.SigmoidTransform().inv, dist.AffineTransform(loc=loc, scale=scale)]
            d = dist.TransformedDistribution(base, transforms)
            ll = d.log_prob(data).sum().item()
            return ll, {'scale': torch.tensor(scale)}, 2
        return -np.inf, {}, 0

    def _get_dist(self, dist_type, params):
        p = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in params.items()}
        if dist_type == 'Normal':
            return dist.Normal(loc=0.0, scale=p['scale'])
        elif dist_type == 'Laplace':
            return dist.Laplace(loc=0.0, scale=p['scale'])
        elif dist_type == 'StudentT':
            return dist.StudentT(df=p['df'], loc=0.0, scale=p['scale'])
        elif dist_type in ['2GMM', '3GMM']:
            mix = dist.Categorical(probs=p['weights'])
            comp = dist.Normal(loc=p['means'], scale=p['scales'])
            return dist.MixtureSameFamily(mix, comp)
        elif dist_type == 'Logistic':
            base = dist.Uniform(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
            transforms = [dist.SigmoidTransform().inv, dist.AffineTransform(loc=0.0, scale=p['scale'])]
            return dist.TransformedDistribution(base, transforms)
        raise ValueError(f"Unknown distribution type: {dist_type}")

    def get_class_labels(self):
        # Return names if provided, else indices
        if self.class_labels_ is not None:
            return list(self.class_labels_)
        return list(self.class_distribution.keys())

    def freeze(self):
        # Method to not update gradients during optimization (if needed)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False


