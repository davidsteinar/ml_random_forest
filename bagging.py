# Author Hesam Pakdaman

import numpy as np
import pandas as pd

'''
Slices the dataframe row-wise and returns a bootstrap sample.
'''
def bootstrap(df):
    all_indx = np.arange(df.shape[0])
    numSamples = 2 * df.shape[0] // 3
    numSamples = 5
    rnd_indx = np.random.randint(0, high = df.shape[0], size=numSamples)
    oob_indx = np.setdiff1d(all_indx, rnd_indx)
    return (df.iloc[rnd_indx], oob_indx)


# Makes k bootstrap samples, contains tuples consisting of the
# bootstrap sample and the list of samples not used.
def bagging(df, k):
    bag = []
    for i in range(k):
        bag.append(bootstrap(df))
    return bag


def testBagging():
    df = pd.read_csv('iris.csv')
    bag = bagging(df, 5)
    for b in bag:
        print(b[0])
        print(b[1])
