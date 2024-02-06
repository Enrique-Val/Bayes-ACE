import pandas as pd
import numpy as np
from pybnesian import hc, CLGNetworkType, SemiparametricBNType

from bayesace.utils import *

df = pd.read_csv("toy-3class.csv")

df["class"] = df["z"].astype('category')
df = df.drop("z", axis=1)

learned = hc(df, bn_type=CLGNetworkType(), operators=["arcs"], score="validated-lik")
learned.num_arcs()

learned.fit(df)

learned_kde = hc(df, bn_type=SemiparametricBNType(), operators=["arcs", "node_type"], score="validated-lik")
learned_kde.fit(df)

print("Slogl:", learned_kde.slogl(df))

print("Succesful run")
