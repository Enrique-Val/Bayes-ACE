import itertools
import math
import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt

import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import numpy as np
import pybnesian as pb

import torch
import torch.nn as nn
import numpy as np


class CLGTorch(nn.Module):
    def __init__(self,
                 topological_order: List[str],
                 node_types: Dict[str, str],
                 bn_dict: Dict[str, Any],
                 continuous_variables: List[str],
                 class_var_name: str = "class"):
        """
        Args:
            continuous_variables: List of names defining the column order of the input tensor.
        """
        super().__init__()
        self.topological_order = topological_order
        self.continuous_variables = continuous_variables
        self.class_var_name = class_var_name

        # 1. Create a map from Name -> Column Index in the input tensor
        self.cont_idx_map = {name: i for i, name in enumerate(continuous_variables)}

        # 2. Setup Discrete Grid (Marginalization logic)
        self.discrete_nodes = []
        self.discrete_cardinalities = {}
        self.discrete_label_map = {}

        for node in topological_order:
            if node_types[node] == "Discrete":
                self.discrete_nodes.append(node)
                labels = list(bn_dict[node].keys())
                self.discrete_label_map[node] = {l: i for i, l in enumerate(labels)}
                self.discrete_cardinalities[node] = len(labels)

        # Build global grid of all discrete combinations
        if self.discrete_nodes:
            ranges = [range(self.discrete_cardinalities[n]) for n in self.discrete_nodes]
            global_grid = list(itertools.product(*ranges))
            self.register_buffer('global_discrete_grid', torch.tensor(global_grid, dtype=torch.long))
            self.num_combos = len(global_grid)
        else:
            self.register_buffer('global_discrete_grid', torch.zeros(1, 0, dtype=torch.long))
            self.num_combos = 1

        # 3. Initialize Nodes with Integer Indices
        self.nodes = nn.ModuleDict()
        for node_name in topological_order:
            node_type = node_types[node_name]
            params = bn_dict[node_name]

            if node_type == "Discrete":
                idx_in_grid = self.discrete_nodes.index(node_name)
                self.nodes[node_name] = DiscreteLogProb(
                    params,
                    self.discrete_label_map[node_name],
                    grid_col_idx=idx_in_grid
                )

            elif node_type == "Gaussian":
                # Only needs indices into the continuous tensor
                self.nodes[node_name] = GaussianLogProb(
                    params,
                    own_idx=self.cont_idx_map[node_name],
                    idx_map=self.cont_idx_map
                )

            elif node_type == "CLG":
                # Needs indices for continuous tensor AND discrete grid
                parent_grid_indices = [self.discrete_nodes.index(p) for p in params['discrete_parents']]
                parent_encoders = {p: self.discrete_label_map[p] for p in params['discrete_parents']}

                self.nodes[node_name] = CLGLogProb(
                    params,
                    parent_encoders,
                    own_idx=self.cont_idx_map[node_name],
                    idx_map=self.cont_idx_map,
                    parent_grid_indices=parent_grid_indices
                )
    # TODO Rework for scenario with multiple discrete nodes
    def get_log_joint(self, x: torch.Tensor):
        """
        Returns log P(X, C) for all C.
        Output Shape: (Batch_Size, Num_Combinations)
        """
        batch_size = x.shape[0]
        total_log_joint = torch.zeros(batch_size, self.num_combos, device=x.device)

        for node_name in self.topological_order:
            node = self.nodes[node_name]

            if isinstance(node, DiscreteLogProb):
                # Shape (Num_Combos,) -> (1, Num_Combos)
                total_log_joint += node(self.global_discrete_grid).unsqueeze(0)

            elif isinstance(node, GaussianLogProb):
                # Shape (Batch,) -> (Batch, 1)
                total_log_joint += node(x).view(batch_size, 1)

            elif isinstance(node, CLGLogProb):
                # Shape (Batch, Num_Combos)
                total_log_joint += node(x, self.global_discrete_grid)

        return total_log_joint

    def forward(self, x: torch.Tensor):
        # P(X) = sum_c P(X, C)
        log_joint = self.get_log_joint(x)
        return torch.logsumexp(log_joint, dim=1)

    def predict_proba(self, x: torch.Tensor):
        """
        Computes P(Class | X) for the given continuous input X.
        Marginalizes out all other discrete variables if they exist.

        Args:
            x: Tensor of shape (Batch_Size, Num_Continuous_Features)

        Returns:
            probs: Tensor of shape (Batch_Size, Num_Classes) summing to 1.
        """
        # 1. Compute Log Joint P(X, All_Discrete_Combos)
        # Shape: (Batch_Size, Num_Combos)
        # This contains P(X, C=0, D=...), P(X, C=1, D=...), etc.
        log_joint_all = self.get_log_joint(x)

        # 2. Check if Class is in discrete nodes (Sanity check)
        if self.class_var_name not in self.discrete_nodes:
            raise ValueError(f"Class variable '{self.class_var_name}' not found in discrete nodes map.")

        # 3. Identify which column in the global grid corresponds to the class variable
        class_grid_idx = self.discrete_nodes.index(self.class_var_name)

        # Extract the column of class labels from the pre-computed grid
        # Shape: (Num_Combos,)
        grid_class_labels = self.global_discrete_grid[:, class_grid_idx]

        # 4. Aggregation: Group columns by Class Label and sum probabilities (logsumexp)
        num_classes = self.discrete_cardinalities[self.class_var_name]

        # List to store log P(X, Class=k) for each k
        log_joint_class = []

        for c in range(num_classes):
            # Create a mask for all grid combinations where Class == c
            # Example: if Combos are [(C=0, D=0), (C=0, D=1), (C=1, D=0)...]
            # Mask for C=0 selects indices [0, 1]
            mask = (grid_class_labels == c)

            # Select the relevant log-joints
            # Shape: (Batch_Size, Num_Matching_Combos)
            log_joint_sub = log_joint_all[:, mask]

            # Marginalize (sum) out the other discrete variables
            # log(Sum(exp(log_p))) -> log P(X, Class=c)
            # Shape: (Batch_Size,)
            log_p_xc = torch.logsumexp(log_joint_sub, dim=1)

            log_joint_class.append(log_p_xc)

        # Stack into shape (Batch_Size, Num_Classes)
        # This represents log P(X, Class) unnormalized
        log_joint_class = torch.stack(log_joint_class, dim=1)

        # 5. Normalize to get P(Class | X)
        # Softmax applies exp() and divides by sum, yielding valid probabilities
        return torch.softmax(log_joint_class, dim=1)


# --- Optimized Components ---

# TODO This will not work if there are multiple discrete nodes hanging from one another. But for numerical classification, this is okay.
class DiscreteLogProb(nn.Module):
    def __init__(self, cpt, label_map, grid_col_idx):
        super().__init__()
        self.grid_col_idx = grid_col_idx
        probs_ordered = [0.0] * len(label_map)
        for label, p in cpt.items():
            probs_ordered[label_map[label]] = p
        self.register_buffer('log_probs', torch.tensor(probs_ordered).log())

    def forward(self, global_grid):
        # Index directly into the cached log_probs
        return self.log_probs[global_grid[:, self.grid_col_idx]]


class GaussianLogProb(nn.Module):
    def __init__(self, params, own_idx, idx_map):
        super().__init__()
        self.own_idx = own_idx
        self.intercept = nn.Parameter(torch.tensor(float(params["intercept"])))
        self.log_scale = nn.Parameter(torch.tensor(float(params["variance"])).sqrt().log())

        # Convert parent names to indices
        parent_names = list(params["cofficients"].keys())
        self.parent_indices = [idx_map[p] for p in parent_names]

        coeffs = [params["cofficients"][p] for p in parent_names]
        self.coeffs = nn.Parameter(torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, x):
        # x shape: (Batch, N_Cont_Vars)
        val = x[:, self.own_idx]

        if not self.parent_indices:
            mu = self.intercept
        else:
            # Slicing is fast and efficient
            parents = x[:, self.parent_indices]
            mu = self.intercept + (parents @ self.coeffs)

        scale = self.log_scale.exp()
        var = scale ** 2
        return -0.5 * ((val - mu) ** 2) / var - self.log_scale - 0.5 * math.log(2 * math.pi)


class CLGLogProb(nn.Module):
    def __init__(self, params, parent_encoders, own_idx, idx_map, parent_grid_indices):
        super().__init__()
        self.own_idx = own_idx
        self.parent_grid_indices = parent_grid_indices

        # Map continuous parents to indices
        cont_parents = params["continuous_parents"]
        self.cont_parent_indices = [idx_map[p] for p in cont_parents]

        # Prepare Parameter Tensors (Intercepts, Variances, Coeffs)
        intercepts = []
        variances = []
        coefficients = []
        self.combo_to_param_idx = {}
        discrete_parents_names = params["discrete_parents"]

        for i, (combo_values, gaussian_params) in enumerate(params["conditionals"].items()):
            combo_indices = tuple(parent_encoders[p][v] for p, v in zip(discrete_parents_names, combo_values))
            self.combo_to_param_idx[combo_indices] = i

            intercepts.append(gaussian_params["intercept"])
            variances.append(gaussian_params["variance"])
            coefficients.append([gaussian_params["cofficients"].get(p, 0.0) for p in cont_parents])

        self.register_buffer("intercepts", torch.tensor(intercepts))
        self.register_buffer("log_scales", torch.tensor(variances).sqrt().log())
        self.register_buffer("coeffs", torch.tensor(coefficients))

    def forward(self, x, global_grid):
        batch_size = x.shape[0]
        num_global_combos = global_grid.shape[0]

        # 1. Map Global Grid -> Param Index
        # (This block effectively 'broadcasts' the local CPD params to the full joint grid)
        # Note: If memory is an issue with massive grids, this mapping can be cached in __init__
        # provided the global grid structure is static (which it is).
        param_indices = torch.zeros(num_global_combos, dtype=torch.long, device=x.device)
        for combo_tuple, param_idx in self.combo_to_param_idx.items():
            mask = torch.ones(num_global_combos, dtype=torch.bool, device=x.device)
            for i, val in enumerate(combo_tuple):
                mask &= (global_grid[:, self.parent_grid_indices[i]] == val)
            if mask.any():
                param_indices[mask] = param_idx

        # 2. Select Params
        grid_intercepts = self.intercepts[param_indices]
        grid_log_scales = self.log_scales[param_indices]
        grid_coeffs = self.coeffs[param_indices]  # (Num_Combos, N_Cont_Parents)

        # 3. Compute Mean
        mu = grid_intercepts.unsqueeze(0).expand(batch_size, -1)

        if self.cont_parent_indices:
            # x subset: (Batch, N_Cont_Parents)
            c_parents = x[:, self.cont_parent_indices]
            # (Batch, N_P) @ (Num_Combos, N_P).T -> (Batch, Num_Combos)
            contribution = c_parents @ grid_coeffs.T
            mu = mu + contribution

        # 4. Compute Log Prob
        val = x[:, self.own_idx].unsqueeze(1)  # (Batch, 1)
        grid_scale = grid_log_scales.exp().unsqueeze(0)
        grid_var = grid_scale ** 2

        return -0.5 * ((val - mu) ** 2) / grid_var - grid_log_scales.unsqueeze(0) - 0.5 * math.log(2 * math.pi)
def process_gaussian_cpd(cpd : pb.LinearGaussianCPD):
    params = {"intercept": cpd.beta[0]}
    params["cofficients"] = {}
    for i, parent in enumerate(cpd.evidence()):
        params["cofficients"][parent] = cpd.beta[i + 1]
    params["variance"] = cpd.variance
    return params

def serialize_clg(bn : pb.CLGNetwork):
    bn_dict = {}
    node_types = {}
    dag = bn.graph()
    topological_order = dag.topological_sort()
    for node in bn.nodes():
        cpd = bn.cpd(node)
        if isinstance(cpd, pb.LinearGaussianCPD):
            node_types[node] = "Gaussian"
            bn_dict[node] = process_gaussian_cpd(cpd)
        elif isinstance(cpd, pb.DiscreteFactor):
            node_types[node] = "Discrete"
            possible_values = cpd.variable_values()
            probs = cpd.probabilities()
            cpt = {val: prob for val, prob in zip(possible_values, probs)}
            bn_dict[node] = cpt
        elif isinstance(cpd, pb.CLinearGaussianCPD):
            node_types[node] = "CLG"
            # First, get the involved discrete parents
            discrete_parents = [p for p in cpd.evidence() if isinstance(bn.cpd(p), pb.DiscreteFactor)]
            discrete_parents_values = {p: bn.cpd(p).variable_values() for p in discrete_parents}
            continuous_parents = [p for p in cpd.evidence() if not isinstance(bn.cpd(p), pb.DiscreteFactor)]
            # Create all combinations of discrete parent values
            from itertools import product
            combinations = list(product(*discrete_parents_values.values()))
            params = {"discrete_parents": discrete_parents, "continuous_parents": continuous_parents, "conditionals": {}}
            for combo in combinations:
                evidence = {p: v for p, v in zip(discrete_parents, combo)}
                assignment = pb.Assignment(evidence)
                conditional_gaussian = cpd.conditional_factor(assignment)
                params["conditionals"][combo] = process_gaussian_cpd(conditional_gaussian)
            bn_dict[node] = params
        else:
            raise ValueError(f"Unsupported CPD type for node {node}: {type(cpd)}")
    return topological_order, node_types, bn_dict