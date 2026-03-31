import torch
from torch import nn

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from bayesace.algorithms.algorithm import Algorithm, ACEResult

from scipy.stats import norm

from bayesace.models.autodiff_proxies.gbn_classifier import serialize_clg, CLGTorch
from bayesace.models.bayesian_network_classifier import BayesianNetworkClassifier

import matplotlib.pyplot as plt

from bayesace.models.conditional_kde import ConditionalKDE
from bayesace.models.conditional_nvp import ConditionalNVP
from bayesace.models.lingam_cat import LingamClassifier


class SGDACE(Algorithm):
    def __init__(self, density_estimator, features, n_vertices=10, lr=0.1, max_epochs=1000, penalty=0,
                 log_likelihood_threshold=-np.inf, posterior_probability_threshold=0.8, chunks=10, continuous = False,
                 trim_features=0, cfx_direction="lower"):
        """
        Args:
            density_estimator: The CLGDensityNetwork (must act as the Torch model).
            features: List of feature names.
            n_vertices: Number of intermediate path points to learn.
            lr: Learning rate for SGD.
            max_epochs: Number of optimization steps.
            penalty: The power penalty applied to the log-likelihood in the line integral.
        """
        # Initialize parent Algorithm class
        super().__init__(density_estimator, features,
                         log_likelihood_threshold=log_likelihood_threshold,
                         posterior_probability_threshold=posterior_probability_threshold)

        self.n_vertices = n_vertices
        self.lr = lr
        self.max_epochs = max_epochs
        self.penalty = penalty
        self.chunks = chunks
        self.logl_scale_factor = 1
        self.continuous = continuous
        self.trim_features = trim_features
        # Only use for continuous counterfactuals
        self.cfx_direction = cfx_direction

        # Ensure model is in eval mode (freeze BN parameters)
        if isinstance(self.density_estimator, nn.Module):
            self.density_estimator.eval()

    def get_device(self):
        if isinstance(self.density_estimator, BayesianNetworkClassifier):
            # TODO to be fixed in next iteration. For now, it works, as we are limiting the implementation to cpu
            return "cpu"
        else:
            return self.density_estimator.device

    def _get_end_probability(self, endpoint: torch.Tensor, target_label):
        """
        Retrieves P(y=target | endpoint).
        Assumes density_estimator has a 'predict_proba' or similar differentiable method.
        If your model only outputs LogL, this needs adaptation to your specific Torch model structure.
        """
        # Checks if the model has a specific prediction method, otherwise assumes forward returns logits/probs
        if hasattr(self.density_estimator, "predict_proba"):
            # Expecting shape (1, n_classes)
            probs = self.density_estimator.predict_proba(endpoint.unsqueeze(0))
            return probs[0, target_label]
        else:
            # Fallback: Raise error or assume the user handles this connection
            # For this code to run, density_estimator MUST support classification queries.
            raise NotImplementedError("density_estimator must implement `predict_proba` for Lagrangian constraints.")

    def _warm_start(self, x_og_tensor, n_samples=2000, target_label=1):
        #print("target_label", target_label)
        """
        Samples points, filters for constraints, and selects the best start
        by evaluating the exact path cost function.
        """
        assert self.n_vertices >= 0

        # In case we input a pandas df, we convert it to tensor here and use it for the rest of the warm start
        # The target label should also be converted to index
        if isinstance(x_og_tensor, pd.DataFrame):
            x_og_tensor = torch.tensor(x_og_tensor[self.features].to_numpy(), dtype=torch.float32, device=self.get_device()).squeeze(0)
            target_label = self.density_estimator.get_class_labels().index(target_label)
        # 1. Sample from the model (Batch)
        if not hasattr(self.density_estimator, "sample"):
            print("Warning: density_estimator has no 'sample' method. Skipping warm start.")
            return None
        metrics = []
        with torch.no_grad():
            candidates = self.density_estimator.sample(n_samples)
            # If they have not been trimmed before
            subset = -1 - self.trim_features
            if self.trim_features > 0 and len(self.features) == x_og_tensor.shape[0]:
                x_og_tensor = x_og_tensor[:-self.trim_features]

            # 2. Filter: Probability Constraint (Batch)
            # We do this in batch for speed, rather than calling _get_end_probability 2000 times
            candidates = torch.tensor(candidates.drop(columns=candidates.columns[subset:]).to_numpy(), dtype=torch.float32)
            if not self.continuous:
                probs = self.density_estimator.predict_proba(candidates)
                p_target = probs[:, target_label]
                valid_prob_mask = p_target >= self.posterior_probability_threshold
            else :
                y_vals = self.density_estimator.continuous_target_argmax(candidates)
                if self.cfx_direction == "lower":
                    valid_prob_mask = y_vals < target_label
                else :
                    valid_prob_mask = y_vals > target_label

            # 3. Filter: Log-Likelihood Constraint (Batch - Optional)
            if self.log_likelihood_threshold > -np.inf:
                log_l = self.density_estimator.logl(candidates)
                valid_logl_mask = log_l >= self.log_likelihood_threshold
                valid_mask = valid_prob_mask & valid_logl_mask
            else:
                valid_mask = valid_prob_mask

            # If no samples meet criteria
            if not valid_mask.any():
                # TODO verbose
                #print(f"Warm Start: 0/{n_samples} samples met constraints. Selecting instance with lowest logl violation.")
                # Proposed fix for Lines 89-91
                # Normalize violations roughly to sum them
                viol_prob_val = torch.clamp(self.posterior_probability_threshold - p_target, min=0)
                viol_logl_val = torch.clamp(self.log_likelihood_threshold - log_l, min=0)

                # Weighted sum (give priority to probability as it's harder to traverse)
                total_violation = viol_prob_val * 10.0 + viol_logl_val
                best_idx = torch.argmin(total_violation)
                best_candidate = candidates[best_idx]
                best_score = self._torch_path_likelihood_length(torch.stack([x_og_tensor, best_candidate], dim=0))
                # TODO verbose
                #print(f"Warm Start: Selected candidate with lowest logl violation. Init Cost: {best_score:.4f}")
                return best_candidate.unsqueeze(0), best_score, 0.0


            # 4. Evaluate Valid Candidates using EXACT Objective
            valid_candidates = candidates[valid_mask]
            best_score = float('inf')
            best_path_params = None

            # 1. Generate the interpolation factors (scalar 0 to 1)
            # Shape: (n_vertices + 1)
            t = torch.linspace(0, 1, self.n_vertices + 2, device=x_og_tensor.device)

            # 2. Reshape for broadcasting
            # Shape: (n_vertices + 1, 1) to multiply against features
            t = t.view(-1, 1)

            # Iterate over valid candidates to find the one with lowest Path Energy
            # (We loop because _torch_path_likelihood_length expects a single path)
            for i in range(len(valid_candidates)):
                candidate = valid_candidates[i]

                # Construct virtual straight line: x_og -> candidate
                virtual_path = x_og_tensor + t * (candidate - x_og_tensor)

                # Calculate Exact Loss
                # We use original=True because that is your chosen objective
                loss = self._torch_path_likelihood_length(virtual_path)
                metrics.append(loss.item())

                if loss < best_score:
                    best_score = loss
                    # Store the parameters (excluding the fixed x_og at index 0)
                    best_path_params = virtual_path[1:].clone()

            #print(f"Warm Start: Selected best path from {len(valid_candidates)} valid candidates. Init Cost: {best_score:.4f}")
            return best_path_params, best_score, np.std(metrics)

    def _torch_path_likelihood_length(self, full_path: torch.Tensor, apply_div=False):
        """
        Vectorized PyTorch implementation of the 'path_likelihood_length' utility.
        Minimizes Integral of (-LogL)^penalty * dl
        """

        # Assert chunks > 1
        assert self.chunks > 1, "Chunks must be greater than 1 for path interpolation."

        # Number of features
        d = full_path.shape[1]

        # Alternative 1: Compute chunks middle points and compute differential
        chunks = self.chunks
        # Iterate over points in full_path and get "chunks" middle points
        expanded_full_path = []
        for i in range(0, full_path.shape[0] - 1):
            start = full_path[i]
            end = full_path[i + 1]
            for j in range(1, chunks + 1):
                alpha = j / (chunks + 1)
                medium_point = (1 - alpha) * start + alpha * end
                expanded_full_path.append(medium_point)
        expanded_full_path = torch.stack(expanded_full_path, dim=0)

        # Compute medium points from expanded full path
        medium_points = (expanded_full_path[:-1] + expanded_full_path[1:]) / 2

        # 2. Separation (dl): Euclidean distance between steps
        # Shape: (N_Vertices + 1,)
        # Alternative 2: Compute distance between every pair of consecutive medium points
        separation = torch.norm(expanded_full_path[1:] - expanded_full_path[:-1], dim=1, p=2)

        # 3. Log Likelihoods (The Density Cost)
        # We use Negative Log Likelihood (NLL) because we want to MINIMIZE cost.
        # High Density = Low Cost.
        log_probs = self.density_estimator.logl(medium_points)
        if apply_div:
            nll_cost = -log_probs/self.logl_scale_factor
        else :
            nll_cost = -log_probs

        # Exponentiate all the points to the penalty (careful with symbols)
        # Store original sign
        sign = torch.sign(nll_cost)
        nll_cost = sign * (torch.abs(nll_cost) ** self.penalty)

        # Multiply every point by the separation
        nll_cost = nll_cost * separation

        return torch.mean(nll_cost)

        # 4. Calculate the two specific integrals

        # Integral A: Density Cost (Energy)
        # "How 'expensive' is the terrain we are crossing?"
        density_integral = torch.mean(nll_cost)

        # Integral B: Euclidean Distance
        # "How long is the path?"
        length_integral = torch.sum(separation**2)

        # 5. Weighted Sum
        # If penalty is HIGH: Path will curve significantly to stay in high-density areas.
        # If penalty is LOW: Path will be a straight line (shortest distance).
        total_loss = density_integral*self.penalty + length_integral
        return total_loss

    def run(self, instance: pd.DataFrame | pd.Series, target_label, verbose = False, initial_guess : torch.Tensor = None,
            ret_norm_loss = False) -> tuple[ACEResult, float] | ACEResult:
        """
        Executes the SGD optimization to find the path.
        """
        # 1. Prepare Data
        # Ensure instance is a DataFrame and extract continuous values in correct order
        if isinstance(instance, pd.Series):
            instance = instance.to_frame().T

        # Freeze the model grads, just in case
        self.density_estimator.freeze()

        # Convert target_label to its index
        target_label = self.density_estimator.get_class_labels().index(target_label)

        # We use self.features to guarantee column order matches the tensor expectation
        x_og_np = instance[self.features].to_numpy().flatten()
        x_og = torch.tensor(x_og_np, dtype=torch.float32, device=self.get_device())
        x_og_copy = x_og.clone()
        # Trim features if needed
        if self.trim_features > 0:
            x_og = x_og[:-self.trim_features]

        # 2. Scale the problem by a constant. This does not change the optimal path, but makes the optimization landscape smoother and more stable.
        with torch.no_grad():
            # We use the log-likelihood of the start point as our unit of measure
            start_logl = self.density_estimator.logl(x_og.unsqueeze(0))
            # Add epsilon to prevent division by zero, take abs
            # Store this as a self variable or pass it
            self.logl_scale_factor = torch.abs(start_logl) + 1e-5

        # Heuristic: Linear initialization + noise to break symmetry
        if initial_guess is None:
            initial_guess, initial_guess_score, sigma = self._warm_start(x_og, n_samples=2000, target_label=target_label)
        else :
            initial_guess_score = self._torch_path_likelihood_length(torch.cat([x_og.unsqueeze(0), initial_guess], dim=0)).item()
            sigma = 0.01 * initial_guess_score

        path_params = nn.Parameter(initial_guess)

        # 3. Optimization Loop
        optimizer = torch.optim.Adam([path_params], lr=self.lr, betas=(0.9, 0.999))

        # Scheduler: Cosine Annealing
        # Starts at 'lr' and decays to 'eta_min' (e.g., 0.001 * lr) by the last epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.lr * 0.01
        )

        final_path = None
        best_loss = np.inf
        best_norm_loss = np.inf

        # ALM Parameters
        # We separate alphas because Probability and Likelihood have different scales
        alpha_prob = 0.0
        alpha_logl = 0.0

        rho_prob = 1.0
        rho_logl = 1.0

        rho_max = 100.0
        rho_multiplier = 1.01

        patience_counter = 0

        for epoch in range(self.max_epochs):
            # If path_params is empty or contains NaN, break and return the best found so far
            if path_params is None or torch.isnan(path_params).any():
                if verbose:
                    print(f"Epoch {epoch}: Path parameters contain NaN. Stopping optimization.")
                break
            optimizer.zero_grad()

            # --- 1. Construct Path & Primal Objective ---
            full_path = torch.cat([x_og.unsqueeze(0), path_params], dim=0)
            endpoint = full_path[-1]

            # Objective: Minimize Path Energy
            raw_path_loss = self._torch_path_likelihood_length(full_path)

            path_loss = (raw_path_loss / initial_guess_score)

            # --- 2. Constraint 1: Target Probability ---
            # g1(x) <= 0
            # If its categorical
            if not self.continuous:
                p_target = self._get_end_probability(endpoint, target_label)
                log_p_target = torch.log(p_target + 1e-9)
                log_threshold = np.log(self.posterior_probability_threshold)

                # Violation 1
                viol_prob = log_threshold - log_p_target

            else :
                y_val = self.density_estimator.continuous_target_argmax(endpoint.unsqueeze(0))
                if self.cfx_direction == "lower":
                    viol_prob = y_val - target_label
                else :
                    viol_prob = target_label - y_val

            # ALM Term 1
            # L_aug = (rho/2) * [max(0, viol + alpha/rho)]^2
            alm_prob = (rho_prob / 2.0) * torch.relu(viol_prob + (alpha_prob / rho_prob)) ** 2

            # --- 3. Constraint 2: Log Likelihood (Optional) ---
            alm_logl = torch.tensor(0.0, device=full_path.device)
            viol_logl_item = -np.inf  # Default to satisfied if constraint doesn't exist

            if self.log_likelihood_threshold > -np.inf:
                end_logl = self.density_estimator.logl(endpoint.unsqueeze(0)).squeeze()

                # Violation 2: Threshold - Actual <= 0 implies Actual >= Threshold
                viol_logl = self.log_likelihood_threshold - end_logl
                viol_logl_item = viol_logl.item()  # For logging and dual updates

                scale_factor = max(1.0, abs(self.log_likelihood_threshold))
                viol_logl = viol_logl / scale_factor


                # ALM Term 2
                alm_logl = (rho_logl / 2.0) * torch.relu(viol_logl + (alpha_logl / rho_logl)) ** 2

            # --- 4. Total Loss & Backward ---
            loss = path_loss + alm_prob+ alm_logl

            loss.backward()

            torch.nn.utils.clip_grad_norm_([path_params], max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # --- 5. Dual Updates (Split) ---
            with torch.no_grad():
                # Update Alpha 1 (Prob) using rho_prob
                alpha_prob_step = alpha_prob + rho_prob * viol_prob.item()
                alpha_prob = max(0.0, alpha_prob_step)

                # Update Alpha 2 (LogL) using rho_logl
                if self.log_likelihood_threshold > -np.inf:
                    alpha_logl_step = alpha_logl + rho_logl * viol_logl_item
                    alpha_logl = max(0.0, alpha_logl_step)

                # --- Update Rhos INDEPENDENTLY ---

                # Update Rho Prob: Only if Probability constraint is violated
                if viol_prob.item() > 0:
                    rho_prob = min(rho_max, rho_prob * rho_multiplier)

                # Update Rho LogL: Only if LogL constraint is active AND violated
                if self.log_likelihood_threshold > -np.inf and viol_logl_item > 0:
                    rho_logl = min(rho_max, rho_logl * rho_multiplier)

            # --- 6. Tracking & Logging ---
            current_path_cost = path_loss.item()

            # Validation Check: Both constraints must be satisfied (violation <= 0)
            # Note: viol_logl_item is -inf if constraint is disabled, so check passes automatically
            is_valid = (viol_prob.item() <= 0) and (viol_logl_item <= 0)

            # 1. Constraints must be satisfied
            if is_valid:
                # 2. Check for convergence (loss hasn't improved significantly)
                # Maintain a 'patience' counter (e.g., stop if no improvement for 50 epochs)
                if best_loss < float('inf') and best_norm_loss - current_path_cost < 1e-4:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter > 200:
                    if verbose:
                        print(f"Converged at epoch {epoch}. Stopping early.")
                    break

            with torch.no_grad():
                raw_path_loss_item = self._torch_path_likelihood_length(full_path, apply_div=False).item()

            if is_valid and raw_path_loss_item <= best_loss:
                best_loss = raw_path_loss_item
                best_norm_loss = current_path_cost
                final_path = path_params.clone().detach()

            if epoch % 50 == 0 and verbose:
                log_msg = (f"Epoch {epoch}: Loss={current_path_cost:.2f} | "
                           f"P(T)={p_target.item():.4f} (Viol={viol_prob.item():.4f}) | "
                           f"A_Prob={alpha_prob:.2f}")

                if self.log_likelihood_threshold > -np.inf:
                    log_msg += f" | LogL Viol={viol_logl_item:.4f} | A_LogL={alpha_logl:.2f}"

                log_msg += f" | Rho_prob={rho_prob:.1f}"
                log_msg += f" | Rho_logl={rho_logl:.1f}"
                print(log_msg)

        # 4. Construct Result
        # Detach and convert to Numpy
        if final_path is None:
            print("Warning: No valid path found that satisfies constraints. Returning None.")
            if ret_norm_loss:
                return ACEResult(None, instance[instance.columns[:-1]], float('inf')), float('inf')
            return ACEResult(None, instance[instance.columns[:-1]], float('inf'))

        if self.trim_features > 0 and isinstance(self.density_estimator, LingamClassifier):
            final_path = self.density_estimator.expand(final_path)
            x_og = x_og_copy

        optimized_path_np = torch.cat([x_og.unsqueeze(0), final_path], dim=0).detach().cpu().numpy()

        path_df = pd.DataFrame(optimized_path_np, columns=self.features)

        # The last point is the counterfactual (even if we ignored the class constraint for now)
        counterfactual = path_df.iloc[-1]

        # Calculate final "distance" using the exact metric
        distance = best_loss

        if ret_norm_loss:
            return ACEResult(counterfactual, path_df, distance), best_norm_loss
        return ACEResult(counterfactual, path_df, distance)

if __name__ == "__main__":
    # --- A. Generate Data ---
    print("--- 1. Generating Data ---")
    np.random.seed(42)
    N = 1000
    C = np.random.randint(0, 2, N)
    X1 = norm(loc=C * 5.0, scale=1.0).rvs(N)
    X2 = 2.0 * X1 + norm(loc=0.0, scale=2.0).rvs(N)
    df = pd.DataFrame({"X1": X1, "X2": X2, "class": C})
    df["class"] = df["class"].astype("category")

    density_est = None
    model_type = "clg"
    if model_type == "clg":
        # --- B. Train CLG ---
        print("--- 2. Training CLG Classifier ---")
        clg_classifier = BayesianNetworkClassifier(network_type="CLG")
        arc_blacklist = [] #[("class", "X2")]
        arc_whitelist = [] #[("class", "X1"), ("X1", "X2")]
        clg_classifier.fit(df.drop(columns=["class"]), df["class"].cat.codes.to_numpy(),
                           initial_structure="empty",
                           training_params={"arc_blacklist": arc_blacklist,
                                            "arc_whitelist": arc_whitelist})
        density_est = clg_classifier
    elif model_type == "nf":
        # --- B. Train Normalizing Flow ---
        print("--- 2. Training Normalizing Flow Classifier ---")
        nf_model = ConditionalNVP()
        nf_model.fit(df.drop(columns=["class"]), df["class"].cat.codes.to_numpy(), hidden_units=10, n_flows=5,
                     layers=2, sam_noise=0.02, batch_size=100)
        density_est = nf_model
    elif model_type == "kde":
        # --- B. Train KDE ---
        print("--- 2. Training KDE Classifier ---")
        kde_model = ConditionalKDE(bandwidth=1.0)
        kde_model.fit(df.drop(columns=["class"]), df["class"].cat.codes.to_numpy())
        density_est = kde_model
    else :
        raise NotImplementedError("Only CLG model is implemented in this example.")


    # Compute mean log-likelihood on training data
    train_logl = density_est.logl(df[["X1", "X2"]])
    mean_train_logl = train_logl.mean().item()
    print(f"Mean Log-Likelihood on Training Data: {mean_train_logl:.4f}")

    #Also, the std
    std_train_logl = train_logl.std().item()
    print(f"Std Dev of Log-Likelihood on Training Data: {std_train_logl:.4f}")

    # --- D. Run SGDACE ---
    print("\n--- 4. Running SGDACE (Path Optimization) ---")

    # Select a starting point (e.g., from Class 0 region)
    # Class 0: X1 ~ 0, X2 ~ 0
    start_instance = pd.DataFrame([[1.0, -1.0]], columns=["X1", "X2"])
    print(f"Start Point:\n{start_instance.to_string(index=False)}")

    # Initialize SGDACE
    sgd_ace = SGDACE(
        density_estimator=density_est,
        features=["X1", "X2"],
        n_vertices=2,
        lr=1e-3,
        max_epochs=1000,
        penalty=10,
        posterior_probability_threshold=0.9,
        log_likelihood_threshold=-6

    )

    # Run Algorithm
    # Note: target_label is ignored in this simplified implementation, but required by signature
    result = sgd_ace.run(instance=start_instance, target_label="1", verbose=True)

    print(f"Final Path Cost (Energy): {result.distance:.4f}")
    print("Learned Path (Coordinates):")
    print(result.path)

    # --- E. Visualization ---
    print("\n--- 5. Visualizing Path on Density Surface ---")
    x1_range = np.linspace(-5, 10, 50)
    x2_range = np.linspace(-10, 25, 50)
    X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
    grid_tensor = torch.tensor(np.column_stack((X1_mesh.ravel(), X2_mesh.ravel())), dtype=torch.float32)
    print(grid_tensor.shape)

    with torch.no_grad():
        log_probs = density_est.logl(grid_tensor)
        Z = log_probs.view(X1_mesh.shape).exp().numpy()
        print(Z.shape)

    plt.figure(figsize=(10, 6))

    # 1. Density Contours
    cp = plt.contourf(X1_mesh, X2_mesh, Z, levels=20, cmap='viridis')
    plt.colorbar(cp, label='Density P(X)')

    # 2. The Learned Path
    path_np = result.path.to_numpy()
    plt.plot(path_np[:, 0], path_np[:, 1], 'r-o', linewidth=3, label='SGDACE Path', markersize=8)
    plt.plot(path_np[0, 0], path_np[0, 1], 'ko', label='Start', markersize=10)
    plt.plot(path_np[-1, 0], path_np[-1, 1], 'rx', label='End', markersize=10, markeredgewidth=3)

    plt.title(f"SGDACE Path Optimization (n_vertices=2)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Check probability at the endpoint
    end_point = torch.tensor(result.path.iloc[-1].to_numpy(), dtype=torch.float32).unsqueeze(0)
    end_prob = density_est.predict_proba(end_point)[0, 1].item()
    print(f"Posterior Probability P(class=1 | endpoint): {end_prob:.6f}")

    # Check logl at the endpoint
    end_logl = density_est.logl(end_point).item()
    print(f"Log-Likelihood at endpoint: {end_logl:.4f}")

    print("Done.")