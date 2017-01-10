# Author Hesam Pakdaman

import numpy as np
import pandas as pd

'''
Slices the dataframe row-wise and returns a bootstrap sample.
'''
def bootstrap(df, size):
    rnd_indx = np.random.randint(0, high=df.shape[0], size=size)
    all_indx = np.arange(df.shape[0])           # indices of the rows
    oob_indx = np.setdiff1d(all_indx, rnd_indx) # oob indices
    return (df.iloc[rnd_indx], oob_indx)


'''
Makes k bootstraps, contains tuples consisting of the
bootstrap and the list of samples not used (oob indices).
'''
def bagging(df, k, size):
    bag = []
    for i in range(k):
        bag.append(bootstrap(df, size))
    return bag


def test_bagging():
    df = pd.read_csv('./datasets/iris.csv')
    bag = bagging(df, 5, 5)
    for b in bag:
        print(b[0])
        print(b[1])
