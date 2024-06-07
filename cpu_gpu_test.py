import random
import csv
import os
import sys

from bayesace.models.multi_bnaf import Arguments, MultiBnaf

from bayesace.models.utils import hill_climbing

sys.path.append(os.getcwd())
import argparse

import pandas as pd
import numpy as np
from pybnesian import hc, CLGNetworkType, SemiparametricBNType, SemiparametricBN, CLGNetwork
# from drawdata import draw_scatter
import matplotlib.pyplot as plt

from bayesace.utils import *
from bayesace.algorithms.bayesace_algorithm import BayesACE
from bayesace.algorithms.face import FACE

from sklearn.preprocessing import StandardScaler
import openml as oml

import multiprocessing as mp

import time




if __name__ == "__main__":
    dataset_id = 44091

    random.seed(0)

    # Load the dataset
    df = get_data(dataset_id)

    # Get the fold indices

    args = Arguments()
    args.layers = 2
    args.hidden_sim= 40
    args.device = "cpu"
    args.load =False
    args.save = False
    args.epochs = 2000
    t0 = time.time()
    mbnaf = MultiBnaf(args, df)
    print("CPU time: ", time.time()-t0)

    args.device = "cuda:0"
    t0 = time.time()
    mbnaf = MultiBnaf(args, df)
    print("GPU time: ", time.time() - t0)
