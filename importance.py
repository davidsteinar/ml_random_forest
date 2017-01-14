# Author Hesam Pakdaman


import pandas as pd
import numpy as np


def permute(df, feat):
    # loop over columns
    df_perm = df.copy()
    values = df_perm.iloc[:,feat].values
    np.random.shuffle(values)
    return df_perm



def test_perm():
    df = pd.read_csv('datasets/biopsy.csv').sample(frac=0.01)
    pcol = 9
    sets_perm = permute(df, pcol)

    print(df)
    print()

    for i in range(len(sets_perm)):
        print(sets_perm[i])
        print()
