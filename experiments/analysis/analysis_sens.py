import os
import pickle

import numpy as np

from bayesace.algorithms.algorithm import ACEResult

results_dir = "../results/exp_eqi_sens2/results"

for noise_dir in os.listdir(results_dir):
    print(noise_dir)
    noise_dir_path = os.path.join(results_dir, noise_dir)
    for alg_id in os.listdir(noise_dir_path)[:3]:
        print(alg_id)
        alg_dir_path = os.path.join(noise_dir_path, alg_id)
        euclidean_diffs = []
        total_dist_per_feature_list = []
        ec_per_feature_list = []
        distance_diff = []
        ec_distance_diff = []
        for instance in range(0,10) :
            # Compute difference between the instance in model 0 and the rest of models
            instance_model_0_path = os.path.join(alg_dir_path, "lingam_0", str(instance)+".pkl")
            with open(instance_model_0_path, "rb") as f:
                res_0 :ACEResult = pickle.load(f)
            res_0_ec_distance = np.sqrt((res_0.path.iloc[0] - res_0.path.iloc[-1]).pow(2).sum())
            for model_id in range(1,31) :
                instance_model_i_path = os.path.join(alg_dir_path, "lingam_"+str(model_id), str(instance)+".pkl")
                with open(instance_model_i_path, "rb") as f:
                    res_i :ACEResult = pickle.load(f)
                # Compute the difference in path length
                euclidean_diff = np.sqrt((res_i.counterfactual - res_0.counterfactual).pow(2).sum())
                euclidean_diffs.append(euclidean_diff)
                total_dist_per_feature = np.abs((res_i.counterfactual - res_0.counterfactual)).mean()
                total_dist_per_feature_list.append(total_dist_per_feature)
                ed_per_feature = euclidean_diff/len(res_i.counterfactual)
                ec_per_feature_list.append(ed_per_feature)
                distance_diff.append((res_i.distance - res_0.distance)**2)
                res_i_ec_distance = np.sqrt(abs(res_i.path.iloc[0] - res_i.path.iloc[-1]).sum())
                ec_distance_diff.append(abs(res_i_ec_distance - res_0_ec_distance) / res_0_ec_distance)
        print("Counterfactual metrics:")
        print(f"Average Euclidean difference for {alg_id} in noise {noise_dir}: {sum(euclidean_diffs)/len(euclidean_diffs)}")
        print(f"Average EC per feature for {alg_id} in noise {noise_dir}: {sum(ec_per_feature_list)/len(ec_per_feature_list)}")
        print(f"Average total distance per feature for {alg_id} in noise {noise_dir}: {sum(total_dist_per_feature_list)/len(total_dist_per_feature_list)}")
        print("Path metrics:")
        print(f"Average squared distance difference for {alg_id} in noise {noise_dir}: {sum(distance_diff)/len(distance_diff)}")
        print(f"Average EC distance difference for {alg_id} in noise {noise_dir}: {sum(ec_distance_diff)/len(ec_distance_diff)}")
        print()

        # TODO normalizar
