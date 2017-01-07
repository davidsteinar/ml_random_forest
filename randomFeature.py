# Author Hesam Pakdaman

import numpy as np
import pandas as pd
from bagging import *

# Predict is an int that specifies the column of the prediction
# value.
def randomIndices(df, predict, m):

    # one variable in df is predicted value
    assert(m < df.shape[1])

    done = False
    features = []
    used = np.zeros(df.shape[1])
    used[predict] = 1

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


def testRandomFeature():
    df = pd.read_csv('iris.csv')
    bag = bagging(df, 5)
    print (df)
    print(randomIndices(bag[0][0], 4, 2))


    for indx, row in df.iterrows():
        print(df.iloc[indx,:][1])
        print(len(df))
