import numpy as np
import lingam
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as dist
from scipy.stats import t, laplace, logistic, norm
from sklearn.mixture import GaussianMixture
import os

# Prevent OpenMP crashes on some systems
os.environ["OMP_NUM_THREADS"] = "1"


# ==========================================
# 1. THE PYTORCH CAUSAL MODULE
# ==========================================
class CausalTorchModule(nn.Module):
    """
    A differentiable PyTorch module representing the learned causal SCM.
    Computes log-likelihoods based on learned structure B and noise distributions.
    """

    def __init__(self, B_matrix, intercepts, noise_config):
        super().__init__()
        # Structure: x = Bx + c + e  =>  e = (I - B)x - c
        self.B = torch.tensor(B_matrix, dtype=torch.float32)
        self.intercepts = torch.tensor(intercepts, dtype=torch.float32)
        self.I = torch.eye(self.B.shape[0])
        self.W = self.I - self.B  # Mixing matrix for inverting to residuals

        # Noise Configuration: Dictionary of {node_index: (dist_type, params_dict)}
        self.noise_config = noise_config

    def forward(self, x):
        """
        Computes Log-Likelihood of input x.
        x shape: (batch_size, n_features) or (n_features,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 1. Recover Residuals: e = (I-B)x - c
        e = x @ self.W.T - self.intercepts

        total_log_prob = 0.0

        # 2. Sum Log-Probs for each node (Local Markov Property)
        for i, (dist_type, params) in self.noise_config.items():
            e_node = e[:, i]

            # Reconstruct distribution on the fly
            if dist_type == 'Normal':
                d = dist.Normal(loc=0.0, scale=params['scale'])
            elif dist_type == 'Laplace':
                d = dist.Laplace(loc=0.0, scale=params['scale'])
            elif dist_type == 'StudentT':
                d = dist.StudentT(df=params['df'], loc=0.0, scale=params['scale'])
            elif dist_type in ['2GMM', '3GMM']:
                # MixtureSameFamily expects (batch, components) for internal broadcasting
                # or specific shapes. simpler here:
                mix = dist.Categorical(probs=params['weights'])
                comp = dist.Normal(loc=params['means'], scale=params['scales'])
                d = dist.MixtureSameFamily(mix, comp)
            elif dist_type == 'Logistic':
                base_distribution = dist.Uniform(0, 1)
                transforms = [dist.SigmoidTransform().inv,
                              dist.AffineTransform(loc=0.0, scale=params['scale'])]
                d = dist.TransformedDistribution(base_distribution, transforms)
            else:
                raise ValueError(f"Unknown distribution: {dist_type}")

            total_log_prob += d.log_prob(e_node)

        # 3. Jacobian Adjustment
        # log p(x) = log p(e) + log|det(I - B)|
        # For DAGs, det(I-B)=1, so log_det=0, but we compute it for generality.
        det_term = torch.slogdet(self.W)[1]

        return total_log_prob + det_term


# ==========================================
# 2. THE LINGAM-TORCH WRAPPER CLASS
# ==========================================
class LingamTorch:
    """
    Wrapper class that:
    1. Fits a DirectLiNGAM model to find structure.
    2. Fits best-match distributions to the residuals.
    3. Exports a CausalTorchModule.
    """

    def __init__(self, random_state=42, prior_knowledge=None):
        self.lingam_model = lingam.DirectLiNGAM(random_state=random_state, prior_knowledge=prior_knowledge)
        self.random_state = random_state
        self.B_ = None
        self.c_ = None
        self.noise_config_ = {}
        self.candidates = ['Normal', 'Laplace', 'StudentT', '2GMM', '3GMM', 'Logistic']

    def fit(self, X_df):
        print(f"--- Fitting Structure (LiNGAM) ---")
        self.lingam_model.fit(X_df)

        # Extract B (Adjacency)
        self.B_ = self.lingam_model.adjacency_matrix_
        print(f"Learned Adjacency Matrix B:\n{self.B_}")

        # Extract Intercepts and Residuals
        # LiNGAM model: X = BX + c + e
        # Therefore: c + e = (I - B)X
        term_c_plus_e = X_df - np.dot(X_df, self.B_.T)
        self.c_ = term_c_plus_e.mean(axis=0).values
        residuals = term_c_plus_e - self.c_

        print(f"Structure learned. Fitting noise distributions to residuals...")
        self._fit_noise_dists(residuals)
        return self

    def _fit_noise_dists(self, residuals_df):
        """Internal method to iterate columns and find best noise fit."""
        n_samples = len(residuals_df)
        tensor_res = torch.tensor(residuals_df.values, dtype=torch.float32)

        for i, col in enumerate(residuals_df.columns):
            data = tensor_res[:, i]
            best_bic = float('inf')
            best_cfg = None

            # Test all candidates
            for dtype in self.candidates:
                ll, params, k_params = self._fit_single_dist(data, dtype)
                # BIC = k*ln(n) - 2LL
                bic = k_params * np.log(n_samples) - 2 * ll

                if bic < best_bic:
                    best_bic = bic
                    best_cfg = (dtype, params)

            self.noise_config_[i] = best_cfg
            print(f"   Node {col}: Selected {best_cfg[0]} (BIC: {best_bic:.1f})")

    def _fit_single_dist(self, data, dist_type):
        """Fits a specific distribution and returns LL, Params, NumParams."""
        if dist_type == 'Normal':
            scale = torch.sqrt((data ** 2).mean())
            ll = dist.Normal(0.0, scale).log_prob(data).sum().item()
            return ll, {'scale': scale}, 1

        elif dist_type == 'Laplace':
            scale = torch.abs(data).mean()
            ll = dist.Laplace(0.0, scale).log_prob(data).sum().item()
            return ll, {'scale': scale}, 1

        elif dist_type == 'StudentT':
            # Use Scipy to get good init params, then eval in Torch
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
            # 3k parameters (mean, var, weight) - 1 constraint on weights sum
        elif "Logistic":
            from scipy.stats import logistic
            loc, scale = logistic.fit(data.numpy(), floc=0)
            # Convert to torch
            base_distribution = dist.Uniform(0, 1)
            transforms = [dist.SigmoidTransform().inv, dist.AffineTransform(loc=loc, scale=scale)]
            logistic = dist.TransformedDistribution(base_distribution, transforms)
            ll = logistic.log_prob(data).sum().item()
            return ll, {'scale': torch.tensor(scale)}, 2

    def to_torch(self):
        """Returns the initialized CausalTorchModule."""
        if self.B_ is None:
            raise ValueError("Model not fitted yet.")
        return CausalTorchModule(self.B_, self.c_, self.noise_config_)

if __name__ == "__main__":
    # Generate synthetic mixed data
    np.random.seed(42)
    n_samples = 5000
    X1 = norm(loc=0,scale=1).rvs(size=n_samples)
    X2 = 2 * X1 + laplace(0,0.5).rvs(size=n_samples)
    Y = X1 + 0.8*X2 + norm(0,1).rvs(size=n_samples)

    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'Y': Y
    })

    # Fit LiM-Torch model
    model = LingamTorch(random_state=42, prior_knowledge=None)
    model.fit(data)

    # Export to PyTorch module
    torch_module = model.to_torch()

    # Test forward pass
    test_sample = torch.tensor([[0.5, 1.0, 1.0]], dtype=torch.float32)
    log_prob = torch_module(test_sample)
    print(f"Log-Probability of test sample: {log_prob.item():.4f}")

    # Train a PyBnesian Gaussian BN on the same data for comparison
    import pybnesian as pb
    bn_gauss = pb.hc(data, score='bic', seed=42, bn_type=pb.GaussianNetworkType(), operators = ["arcs"])
    bn_gauss.fit(data)

    # Test log-likelihood  on the same sample
    test_sample_df = pd.DataFrame(test_sample.numpy(), columns=['X1', 'X2', 'Y'])
    ll_bn = bn_gauss.logl(test_sample_df)[0]
    print(f"PyBnesian Gaussian BN Log-Likelihood of test sample: {ll_bn:.4f}")

    # Print arcs of BN
    print("Learned arcs in PyBnesian Gaussian BN:")
    for arc in bn_gauss.arcs():
        print(f"  {arc[0]} -> {arc[1]}")