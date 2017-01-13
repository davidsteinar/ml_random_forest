# Author Hesam Pakdaman


import pandas as pd
import numpy as np


def permute(df, pcol):
    sets_perm = []
    # loop over columns
    for i in range(df.shape[1]):
        if(i is not pcol):
            values = df.values[:,i]
            np.random.shuffle(values)
            df_perm = df.copy()
            df_perm.iloc[:,i] = values
            sets_perm.append(df_perm)
    return sets_perm






def test_perm():
    df = pd.read_csv('datasets/iris.csv').sample(frac=0.1)
    pcol = 4
    sets_perm = permute(df, pcol)

    print(df)
    print()

    for i in range(len(sets_perm)):
        print(sets_perm[i])
        print()
