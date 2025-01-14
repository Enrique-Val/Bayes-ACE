import pybnesian as pb
import os
from bayesace.utils import *


def test_likelihood(x_cfx: pd.DataFrame, bn):
    assert round(log_likelihood(x_cfx.drop("class", axis=1), bn)[0], 2) == -13.76


def test_accuracy(x_cfx: pd.DataFrame, y_og: str | list, bn):
    assert round(bn.posterior_probability(x_cfx.drop("class", axis=1), y_og)[0], 2) == 0.13


def test_predict_class(data: pd.DataFrame, bn):
    assert (bn.predict_proba(data.values, output="pandas").map(np.log).map(round2).values == np.array([[-19.51, -0., -96.85],
                                                                                  [-24.03, -131.18, -0.],
                                                                                  [-18.63, -147.13, -0.],
                                                                                  [-20.07, -108.58, -0.],
                                                                                  [-30.74, -0., -112.18]])).all()


def test_path(vertex_array: np.ndarray, chunks=2):
    assert (path(vertex_array, chunks)[0].round(2) == np.array([[236.07, 414.81],
       [260.21, 292.73],
       [284.35, 170.65 ],
       [308.49,  48.57]])).all()


def round2(x):
    return round(x, 2)


if __name__ == "__main__":
    assert pb.__version__ == "0.4.3"
    np.random.seed(0)

    file = os.path.dirname(__file__) + "/test_dataset.csv"
    df = pd.read_csv(file)
    df["class"] = df["z"].astype('category')
    df = df.drop("z", axis=1)

    bn = pb.hc(df, bn_type=pb.CLGNetworkType(), operators=["arcs"], score="validated-lik", seed=0)
    bn.fit(df)
    x_1 = df.iloc[[0]]
    x_2 = df.iloc[[200]]
    x_vertex = df.iloc[[0,200]].drop("class", axis = 1).to_numpy()
    sample = bn.sample(5, seed=0).to_pandas()

    test_likelihood(x_1, bn)
    test_accuracy(x_2, "a", bn)
    test_predict_class(sample,bn)
    test_path(x_vertex, chunks=4)


    print("Utils tested succesfully")
