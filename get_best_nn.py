import pandas as pd

datasets_ids = [44091, 44122, 44123, 44127, 44130]

print("Selecting by best logl")
best_nns = {}
for i in datasets_ids:
    print("Dataset", i)
    data = pd.read_csv('./results/exp_cv/data' + str(i) + '.csv', index_col=0)
    mini_df = pd.DataFrame(columns=["logl", "brier", "auc"], index = ["CLG", "Best"])
    mini_df.loc["CLG","logl"] = round(data.loc["mean_logl"]["CLG"],2)
    mini_df.loc["CLG","brier"] = round(data.loc["mean_brier"]["CLG"],2)
    mini_df.loc["CLG","auc"] = round(data.loc["auc_mean"]["CLG"],2)
    data = data.drop("CLG",axis=1)
    data = data.drop("SP",axis=1)
    briers = data.loc["mean_logl"]
    min_model = briers.idxmax()
    best_nns[i] = min_model
    mini_df.loc["Best","logl"] = round(data.loc["mean_logl"][min_model], 2)
    mini_df.loc["Best","brier"] = round(data.loc["mean_brier"][min_model], 2)
    mini_df.loc["Best","auc"] = round(data.loc["auc_mean"][min_model], 2)
    print(mini_df)
    print()

print()

print("Selecting by best brier")
best_nns = {}
for i in datasets_ids:
    print("Dataset", i)
    data = pd.read_csv('./results/exp_cv/data' + str(i) + '.csv', index_col=0)
    mini_df = pd.DataFrame(columns=["logl", "brier", "auc"], index = ["CLG", "Best"])
    mini_df.loc["CLG","logl"] = round(data.loc["mean_logl"]["CLG"],2)
    mini_df.loc["CLG","brier"] = round(data.loc["mean_brier"]["CLG"],2)
    mini_df.loc["CLG","auc"] = round(data.loc["auc_mean"]["CLG"],2)
    data = data.drop("CLG",axis=1)
    data = data.drop("SP",axis=1)
    briers = data.loc["mean_brier"]
    min_model = briers.idxmin()
    best_nns[i] = min_model
    mini_df.loc["Best","logl"] = round(data.loc["mean_logl"][min_model], 2)
    mini_df.loc["Best","brier"] = round(data.loc["mean_brier"][min_model], 2)
    mini_df.loc["Best","auc"] = round(data.loc["auc_mean"][min_model], 2)
    print(mini_df)
    print()

print()

print("Selecting by best AUC")
best_nns = {}
for i in datasets_ids:
    print("Dataset", i)
    data = pd.read_csv('./results/exp_cv/data' + str(i) + '.csv', index_col=0)
    mini_df = pd.DataFrame(columns=["logl", "brier", "auc"], index = ["CLG", "Best"])
    mini_df.loc["CLG","logl"] = round(data.loc["mean_logl"]["CLG"],2)
    mini_df.loc["CLG","brier"] = round(data.loc["mean_brier"]["CLG"],2)
    mini_df.loc["CLG","auc"] = round(data.loc["auc_mean"]["CLG"],2)
    data = data.drop("CLG",axis=1)
    data = data.drop("SP",axis=1)
    briers = data.loc["auc_mean"]
    min_model = briers.idxmax()
    best_nns[i] = min_model
    mini_df.loc["Best","logl"] = round(data.loc["mean_logl"][min_model], 2)
    mini_df.loc["Best","brier"] = round(data.loc["mean_brier"][min_model], 2)
    mini_df.loc["Best","auc"] = round(data.loc["auc_mean"][min_model], 2)
    print(mini_df)
    print()

print()