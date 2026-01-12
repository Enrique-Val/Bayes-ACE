import numpy as np
import lingam
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as dist
from dask.array.random import laplace
from scipy.stats import t, laplace, logistic
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
import os

from bayesace.models.bayesian_network_classifier import BayesianNetworkClassifier

# Prevent OpenMP crashes on some systems
os.environ["OMP_NUM_THREADS"] = "1"


# ==========================================
# 1. THE LIM PYTORCH CAUSAL MODULE
# ==========================================
class LiMCausalTorchModule(nn.Module):
    """
    A differentiable PyTorch module representing a LiM (Linear Mixed) Causal Model.
    - Variables 0 to n-2: Continuous (Standard LiNGAM)
    - Variable n-1: Discrete/Binary (Logistic Model)
    """

    def __init__(self, B_matrix, intercepts, noise_config, discrete_idx):
        super().__init__()
        self.B = torch.tensor(B_matrix, dtype=torch.float32)
        self.intercepts = torch.tensor(intercepts, dtype=torch.float32)

        # Identity matrix for continuous residual calculation
        self.I = torch.eye(self.B.shape[0])
        self.W = self.I - self.B

        self.noise_config = noise_config
        self.discrete_idx = discrete_idx  # The index of the binary variable (n-1)

    def forward(self, x):
        """
        Computes Log-Likelihood of input x.
        x shape: (batch_size, n_features) or (n_features,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Handle single sample

        total_log_prob = 0.0

        # --- PART A: CONTINUOUS NODES (Standard LiNGAM) ---
        # 1. Recover residuals for ALL nodes first: e = (I-B)x - c
        # (We only use the residuals for the continuous ones)
        e_all = x @ self.W.T - self.intercepts

        # 2. Sum log-probs for continuous nodes
        for i, (dist_type, params) in self.noise_config.items():
            if i == self.discrete_idx:
                continue  # Skip the discrete node here

            e_node = e_all[:, i]

            # Reconstruct distribution on the fly
            if dist_type == 'Normal':
                d = dist.Normal(loc=0.0, scale=params['scale'])
            elif dist_type == 'Laplace':
                d = dist.Laplace(loc=0.0, scale=params['scale'])
            elif dist_type == 'StudentT':
                d = dist.StudentT(df=params['df'], loc=0.0, scale=params['scale'])
            elif dist_type in ['2GMM', '3GMM']:
                mix = dist.Categorical(probs=params['weights'])
                comp = dist.Normal(loc=params['means'], scale=params['scales'])
                d = dist.MixtureSameFamily(mix, comp)
            else:
                raise ValueError(f"Unknown distribution: {dist_type}")

            total_log_prob += d.log_prob(e_node)

        # 3. Jacobian Adjustment (Only relevant for continuous part)
        # For DAGs, det(I-B)=1, log_det=0. We add it for completeness/correctness.
        det_term = torch.slogdet(self.W)[1]
        total_log_prob += det_term

        # --- PART B: DISCRETE NODE (Binary Logistic) ---
        # We don't use residuals here. We use the raw linear prediction "score".
        # Score z = (Parents * Coefs) + Intercept
        # Note: Bx + c is exactly the prediction from parents

        # Extract the row of B corresponding to the discrete node
        # We need (Batch, N) @ (N, 1) -> (Batch, 1)
        # B_row shape: (1, n_features)

        # Calculate Bx for all nodes (Predictions)
        preds = x @ self.B.T + self.intercepts

        # Get the score (z) for the discrete node only
        z = preds[:, self.discrete_idx]

        # Get the observed values (0 or 1)
        obs = x[:, self.discrete_idx]

        # Log-Likelihood for Binary:
        # If obs=1: log(sigmoid(z))
        # If obs=0: log(1 - sigmoid(z))
        # We use LogSigmoid for numerical stability
        log_sigmoid = nn.LogSigmoid()

        # term 1: obs * log(sig(z))
        # term 0: (1-obs) * log(1 - sig(z)) = (1-obs) * log(sig(-z))
        term_1 = obs * log_sigmoid(z)
        term_0 = (1 - obs) * log_sigmoid(-z)

        total_log_prob += (term_1 + term_0)

        return total_log_prob


# ==========================================
# 2. THE LIM-TORCH WRAPPER CLASS
# ==========================================
class LiMTorch:
    """
    Wrapper class for Mixed Data (Continuous + Last Variable Discrete)
    1. Fits LiM model (Lingam with mixed data support).
    2. Fits noise distributions for continuous vars.
    3. Exports a LiMCausalTorchModule.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.B_ = None
        self.c_ = None
        self.noise_config_ = {}
        self.candidates = ['Normal', 'Laplace', 'StudentT', '2GMM', '3GMM']
        self.discrete_idx = None

    def fit(self, X_df):
        n_features = X_df.shape[1]
        self.discrete_idx = n_features - 1  # Assumption: Last var is discrete

        print(f"--- Fitting Structure (LiM) ---")
        # We assume X_df is a dataframe. Last column is discrete.
        # LiM requires specifying which columns are discrete via their indices?
        # No, lingam.LiM takes distinct_counts argument usually, or just handles it if passed correctly.
        # However, standard lingam.LiM() usage:
        # model = lingam.LiM()
        # model.fit(X, distinct_counts={self.discrete_idx: 2})

        # NOTE: If your installed version of 'lingam' doesn't support LiM class directly,
        # you might need to use Resit or similar, but assuming 'lingam' package has LiM:

        self.model = lingam.LiM()
        # distinct_counts is a dict: {column_index: number_of_categories}
        # We assume binary (2 categories) for the last column
        # Array that marks continuous vs discrete
        dis_con = np.ones(n_features, dtype=int)
        dis_con[self.discrete_idx] = 0  # 0 for discrete
        dis_con = dis_con.reshape(1,-1)
        X_np = X_df.to_numpy()
        self.model.fit(X_np, dis_con)
        self.B_ = self.model.adjacency_matrix_

        # --- RECOVER PARAMETERS ---

        # 1. Intercepts
        # For continuous vars: c = mean(x - Bx)
        # For discrete vars: The intercept is implicit in the logistic score
        # But we need to extract it carefully.

        # LiM doesn't always expose intercepts easily in .intercept_.
        # We will manually recalculate them to be safe and consistent.

        X_np = X_df.values
        n = X_np.shape[0]
        self.c_ = np.zeros(n_features)

        # A. Continuous Intercepts (0 to n-2)
        # Calculate raw residuals (including intercept)
        term_continuous = X_np[:, :-1] - (X_np @ self.B_.T)[:, :-1]
        self.c_[:-1] = term_continuous.mean(axis=0)

        residuals_continuous = term_continuous - self.c_[:-1]

        # B. Discrete Intercept (n-1)
        # We need to run a Logistic Regression on the identified parents to get the exact intercept
        # consistent with the graph structure found by LiM.
        parents_indices = np.where(self.B_[self.discrete_idx, :] != 0)[0]

        if len(parents_indices) > 0:
            lr = LogisticRegression(solver='lbfgs', C=1e10, fit_intercept=True)  # Unregularized
            lr.fit(X_np[:, parents_indices], X_np[:, self.discrete_idx])
            self.c_[self.discrete_idx] = lr.intercept_[0]
            # Optional: Refine B coefficients for this row using the LR result for better alignment
            # self.B_[self.discrete_idx, parents_indices] = lr.coef_[0]
        else:
            # Root node binary
            p_1 = X_np[:, self.discrete_idx].mean()
            # intercept is log odds
            self.c_[self.discrete_idx] = np.log(p_1 / (1 - p_1))

        # --- FIT NOISE (Continuous Only) ---
        print(f"Structure learned. Fitting noise distributions to continuous residuals...")

        # Create a temp dataframe for continuous residuals
        res_df = pd.DataFrame(residuals_continuous, columns=X_df.columns[:-1])
        self._fit_noise_dists(res_df)

        return self

    def _fit_noise_dists(self, residuals_df):
        """Fits distributions to continuous residuals."""
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
            print(f"   Node {col} (Cont): Selected {best_cfg[0]} (BIC: {best_bic:.1f})")
            print(f"      Params: {best_cfg[1]}")

    def _fit_single_dist(self, data, dist_type):
        """(Same as previous code)"""
        if dist_type == 'Normal':
            scale = torch.sqrt((data ** 2).mean())
            ll = dist.Normal(0.0, scale).log_prob(data).sum().item()
            return ll, {'scale': scale}, 1
        elif dist_type == 'Laplace':
            scale = torch.abs(data).mean()
            ll = dist.Laplace(0.0, scale).log_prob(data).sum().item()
            return ll, {'scale': scale}, 1
        elif dist_type == 'StudentT':
            params = t.fit(data.numpy(), floc=0)
            df, scale = params[0], params[2]
            ll = dist.StudentT(df=df, loc=0.0, scale=scale).log_prob(data).sum().item()
            return ll, {'df': torch.tensor(df), 'scale': torch.tensor(scale)}, 2
        elif 'GMM' in dist_type:
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

    def to_torch(self):
        """Returns the initialized LiMCausalTorchModule."""
        if self.B_ is None:
            raise ValueError("Model not fitted yet.")
        return LiMCausalTorchModule(self.B_, self.c_, self.noise_config_, self.discrete_idx)

# Create a main block for quick testing
if __name__ == "__main__":
    # Generate synthetic mixed data
    np.random.seed(42)
    n_samples = 1000
    X1 = t(df=4,loc=0,scale=1).rvs(size=n_samples)
    Y = ((X1 + logistic(0,1).rvs(size=n_samples)) > 0).astype(int)
    X2 = 0.5 * Y + laplace(loc=0,scale=1).rvs(size=n_samples)

    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'Y': Y
    })

    # Fit LiM-Torch model
    model = LiMTorch(random_state=42)
    model.fit(data)

    # Export to PyTorch module
    torch_module = model.to_torch()

    # Test forward pass
    test_sample = torch.tensor([[0.5, 1.0, 1.0]], dtype=torch.float32)
    log_prob = torch_module(test_sample)
    print(f"Log-Probability of test sample: {log_prob.item():.4f}")

    # For contrast, learn a conditional Bayesian network classifier on the same data
    clf = BayesianNetworkClassifier(network_type="CLG")
    data_clg = data.copy()
    data_clg['Y'] = data_clg['Y'].astype('category')
    clf.fit(data[['X1', 'X2']], data['Y'], training_params={"score": "bic", "seed": 42})
    # Compute log-likelihood of the same test sample
    test_sample_clf = pd.DataFrame({'X1': [0.5], 'X2': [1.0], 'Y': ["1"]})
    logl_clf = clf.logl(test_sample_clf[['X1', 'X2', 'Y']])
    print(f"CLG Classifier Log-Probability of test sample: {logl_clf.item():.4f}")

    # Print arcs
    for arc in clf.bayesian_network.arcs():
        print(f"Arc: {arc[0]} -> {arc[1]}")