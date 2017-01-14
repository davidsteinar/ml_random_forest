# Author: David Steinar, Hesam Pakdaman

import pandas as pd
import numpy as np
import time



def gini(f):
    counts = np.zeros(100)       # fast solution to assume 100 classes
    for value in f:
        counts[value] += 1
    return 1 - sum(np.square(counts / len(counts)))



def gini_split(G, df, column_ids, predict):

    if(G < 0.02 or df.shape[0] < 3):
        return -1, None, None, None, None, None

    gini_vals = []
    gini_splits = []

    best_gini = None
    total_gini = 0

    for i in column_ids:

        # get feature column
        x = df.iloc[:, i]
        N = len(x)

        min_val = x.min()
        diff = x.max() - min_val

        k = 4

        # split into k-1 pieces
        for j in range(1, k):

            split_value = (j * diff / k) + min_val

            df_l = df[x < split_value]
            df_r = df[x >= split_value]

            N1 = len(df_l)
            N2 = len(df_r)
            N = N1 + N2

            # calculate weighted gini for the split
            G1 = N1 / N * gini(df_l.iloc[:,predict])
            G2 = N2 / N * gini(df_r.iloc[:,predict])

            gini_gain = G - (G1 + G2)

            if(best_gini is None or best_gini < gini_gain):
                best_gini = gini_gain
                best_split = split_value
                best_feature = i
                best_dfl = df_l
                best_dfr = df_r
            total_gini += G1 + G2


    if (gini_gain <= 0):
        return -1, None, None, None, None, None
    else:
        return best_feature, best_split, best_dfl, best_dfr, G1, G2



def test_gini():
    df = pd.read_csv('./datasets/iris.csv')
    pcol = 4
    G = gini(df.iloc[:, pcol])
    start = time.clock()
    gini_split(G, df, [2, 0], pcol)
    print(time.clock() - start)
