# Author Hesam Pakdaman

import numpy as np
import pandas as pd
from bagging import *

'''
Assumes the dataset is stripped of  unecessary info. pcol is an int that
specifies  the column  of  the prediction  values. m  is  the number  of
features that we want. Returns a list of indices pertaining to features.
'''
def rand_indx(df, pcol, m):

    # one variable in df is predictive
    assert(m < df.shape[1])

    features = []
    used = np.zeros(df.shape[1])
    used[pcol] = 1

    done = False
    # pick random indices, check that they are unique
    # and not the predicted variable
    while(not done):
        feature_indx = np.random.randint(low=0, high=df.shape[1])
        if(not used[feature_indx]):
            used[feature_indx] = 1
            features.append(feature_indx)
        if(len(features) == m):
            done = True
    return np.asarray(features)


def testing():
    df = pd.read_csv('./datasets/iris.csv')

    # 5 bootstaps, with size of 5 rows
    bag = bagging(df, 5, 5)
    print (df)
    print(rand_indx(bag[0][0], 4, 2))

    for indx, row in df.iterrows():
        print(df.iloc[indx,:][1])
        print(len(df))
