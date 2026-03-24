import argparse
import os
import pickle
from collections import OrderedDict

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Integer, Real

from bayesace import ConditionalDE
from bayesace.models.conditional_normalizing_flow import ConditionalNF

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_id", nargs='?', default=44089, type=int)
    parser.add_argument("--model", nargs='?', default="nf", type=str, choices=["clg", "nf"])
    parser.add_argument('--cv_dir', nargs='?', default='./results/exp_cv_2/', type=str)
    parser.add_argument('--results_dir', nargs='?', default='./results/mia/', type=str)
    args = parser.parse_args()

    results_dir = os.path.join(args.results_dir, str(args.dataset_id))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    cv_dir = os.path.join(args.cv_dir, str(args.dataset_id))

    data_train = pd.read_csv(os.path.join(cv_dir, "resampleddata_" + str(args.dataset_id) + ".csv"), index_col=0)
    data_attack = pd.read_csv(os.path.join(results_dir, "data_attack.csv"))
    features = list(data_train.columns[:-1])
    target_val = data_train.columns[-1]

    # Open density estimator
    if args.model == "clg":
        de_path = os.path.join(cv_dir, "clg_" + str(args.dataset_id) + ".pkl")
    else :
        de_path = os.path.join(cv_dir, "nf_" + str(args.dataset_id) + ".pkl")
    with open(de_path, "rb") as f:
        de: ConditionalDE = pickle.load(f)

    # 1. Extract Signals
    train_logl = de.logl(data_train[features])
    attack_logl = de.logl(data_attack[features])

    if isinstance(de, ConditionalNF) :
        # Convert to numpy
        train_logl = train_logl.cpu().detach().numpy()
        attack_logl = attack_logl.cpu().detach().numpy()

    train_proba = de.predict_proba(data_train[features], output= "numpy")[:, 0]
    attack_proba = de.predict_proba(data_attack[features], output="numpy")[:, 0]

    train_conf = np.maximum(train_proba, 1 - train_proba)
    attack_conf = np.maximum(attack_proba, 1 - attack_proba)

    # 2. Setup Base MIA Arrays
    y_true = np.concatenate([np.ones(len(train_logl)), np.zeros(len(attack_logl))])
    scores_logl = np.concatenate([train_logl, attack_logl])
    scores_conf = np.concatenate([train_conf, attack_conf])

    # 3. Build Scenarios
    # Scenario 1: Log-likelihood + Confidence Only
    X_scenario_1 = np.column_stack((scores_logl, scores_conf))

    # Scenario 2: Log-likelihood + Confidence + Original Features
    # Concatenate the original feature sets to match the order of y_true
    combined_features = pd.concat([data_train[features], data_attack[features]]).values
    X_scenario_2 = np.column_stack((scores_logl, scores_conf, combined_features))

    # 4. Train Meta-Classifiers
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gbc', GradientBoostingClassifier(random_state=42))
    ])

    # Define the search space using skopt's specific space classes
    # Notice we use 'log-uniform' for the learning rate, which is a Bayesian best practice
    search_spaces = {
        'gbc__n_estimators': Integer(50, 500),
        'gbc__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'gbc__max_depth': Integer(3, 15),
        'gbc__min_samples_split': Integer(2,10),
        'gbc__subsample': Real(0.8, 1.0, prior='uniform')
    }

    # Setup BayesSearchCV
    # It looks and acts exactly like GridSearchCV or RandomizedSearchCV
    bayes_search = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_spaces,
        n_iter=50,  # Number of Bayesian iterations
        cv=10,
        scoring=['roc_auc', 'accuracy'],  # Calculate both metrics at once
        refit='roc_auc',  # Optimize for AUC
        n_jobs=-1,
        random_state=42
    )


    def get_best_scores(search_obj) -> tuple[float, float, OrderedDict] :
        """Helper to extract the multi-metric scores for the best performing model."""
        best_index = search_obj.best_index_
        best_auc = search_obj.cv_results_['mean_test_roc_auc'][best_index]
        best_acc = search_obj.cv_results_['mean_test_accuracy'][best_index]
        best_params : OrderedDict = search_obj.cv_results_['params'][best_index]
        return best_auc, best_acc, best_params


    # Evaluate Scenario 1
    print("Running BayesSearchCV for Scenario 1...")
    bayes_search.fit(X_scenario_1, y_true)
    auc_s1, acc_s1, bp_s1 = get_best_scores(bayes_search)

    # Evaluate Scenario 2
    print("Running BayesSearchCV for Scenario 2...")
    bayes_search.fit(X_scenario_2, y_true)
    auc_s2, acc_s2, bp_s2 = get_best_scores(bayes_search)

    # 5. Output and Save Results
    print(f"\n--- Combined MIA Results for Dataset {args.dataset_id} ---")
    print(f"Scenario 1 (Signals Only)       -> AUC: {auc_s1:.4f} | CV Accuracy: {acc_s1:.4f}")
    print(f"Scenario 2 (Signals + Features) -> AUC: {auc_s2:.4f} | CV Accuracy: {acc_s2:.4f}\n")

    # Save to CSV
    results_df = pd.DataFrame({
        'Metric': ['AUC', 'Accuracy'] + list(bp_s1.keys()),
        'Scenario_1_Signals_Only': [auc_s1, acc_s1] + list(bp_s1.values()),
        'Scenario_2_Signals_And_Features': [auc_s2, acc_s2] + list(bp_s2.values())
    })

    results_out_path = os.path.join(results_dir, "mia_results.csv")
    results_df.to_csv(results_out_path, index=False)
    print(f"Results saved to: {results_out_path}")