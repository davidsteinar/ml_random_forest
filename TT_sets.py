# Author Pantelis

import pandas as pd

'''
Will split dataset df into training set with fraction
q and train set with fraction 1-q.
'''
def TT_sets(df, q):
    df_train = df.sample(frac = q)
    df_test = df.drop(train_set.index[:])
    return train_set, test_set
