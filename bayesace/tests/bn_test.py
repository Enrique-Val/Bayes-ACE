import pybnesian as pb
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler


def test_gaussian_network(df):
    bn_arcs = [('x', 'y'), ('y', 'class'), ('x', 'class')]
    bn = pb.hc(df, bn_type=pb.GaussianNetworkType(), operators=["arcs"], score="bic", seed=0)
    bn.fit(df)
    sample = bn.sample(100, ordered=True, seed=0).to_pandas()
    assert bn.arcs() == bn_arcs
    assert round(bn.slogl(sample), 2) == -365.65


def test_clg_network(df):
    bn_arcs = [('x', 'y'), ('class', 'x'), ('class', 'y')]
    bn = pb.hc(df, bn_type=pb.CLGNetworkType(), operators=["arcs"], score="validated-lik", seed=0)
    bn.fit(df)
    sample = bn.sample(100, ordered=True, seed=0).to_pandas()
    assert bn.arcs() == bn_arcs
    assert round(bn.slogl(sample), 2) == -183.85


def test_spbn_network(df):
    bn_arcs = [('y', 'x'), ('class', 'x'), ('class', 'y')]
    bn = pb.hc(df, bn_type=pb.SemiparametricBNType(), operators=["arcs", "node_type"], score="validated-lik", seed=0)
    bn.fit(df)
    sample = bn.sample(100, ordered=True, seed=0).to_pandas()
    assert bn.arcs() == bn_arcs
    assert round(bn.slogl(sample), 2) == -166.66


if __name__ == "__main__":
    assert pb.__version__ == "0.4.3"
    np.random.seed(0)
    file = os.path.dirname(__file__) + "/test_dataset.csv"
    df = pd.read_csv(file)
    df["class"] = df["z"].astype('category')
    df = df.drop("z", axis=1)

    df_num = df.copy().drop("class", axis=1)

    df_num["class"] = [ord(i) - 97 for i in df["class"]]
    feature_columns = [i for i in df.columns if i != "class"]
    df[feature_columns] = StandardScaler().fit_transform(df[feature_columns].values)
    df_num[:] = StandardScaler().fit_transform(df_num.values)

    test_gaussian_network(df_num)
    test_clg_network(df)
    test_spbn_network(df)
    print("BN tested succesfully")
