# Author Pantelis

import pandas as pd

'''
Will split dataset df into training set with fraction
q and train set with fraction 1-q.
'''
def TT_sets(df, q):
    df_train = df.sample(frac=q)
    df_test = df.drop(df_train.index[:])
    return df_train, df_test
