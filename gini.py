#======= Pantelis ========#

import pandas as pd
import numpy as np
from random import shuffle

def gini_index(df, predict):

    d_class = df.iloc[:, k].astype("category")

    count = 0
    prob = 0
    total = 0

    d_class.cat.codes.align

    for i in range(d_class.cat.codes[len(d_class)-1]+1):
        for j in range(len(d_class)-1):
            if  d_class.cat.codes[j] == i:
                count += 1
        prob = (count / (len(d_class)-1))**2
        total += prob
        count = 0

    g = 1 - total
    return g


def testGini():
    df = pd.read_csv('iris.csv')
    gini = gini_index(df)
    print("gini index =", gini)
