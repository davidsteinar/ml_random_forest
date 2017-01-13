# Author: David Steinar, Hesam Pakdaman

import pandas as pd
import numpy as np
import time


def gini(f):
    categories = f.unique()
    p = np.zeros(len(categories),dtype=int)
    for value in f:
        loc = np.where(categories == value)
        p[loc[0][0]] += 1
    p = p/len(f)
    return 1-sum(np.square(p))

def gini_split(G, df, column_ids, predict):

    if(G < 0.05 or df.shape[0] < 10):
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

        k = 3

        # split into k-1 pieces
        for j in range(1, k):

            split_value = j * diff // k + min_val

            df_l = df[x < split_value]
            df_r = df[x >= split_value]

            N1 = len(df_l)
            N2 = len(df_r)
            N = N1 + N2

            # calculate weighted gini for the split
            G1 = N1 / N * gini(df_l.ix[:,predict])
            G2 = N2 / N * gini(df_r.ix[:,predict])

            gini_gain = G - (G1 + G2)

            if(best_gini is None or best_gini < gini_gain):
                best_gini = gini_gain
                best_split = split_value
                best_feature = i
            total_gini += G1 + G2


    if (gini_gain <= 0):
        return -1, None, None, None, None, None
    else:
        df_l = df[df.iloc[:, best_feature] < best_split]
        df_r = df[df.iloc[:, best_feature] >= best_split]
        return best_feature, best_split, df_l, df_r, G1, G2



def test_gini():
    df = pd.read_csv('./datasets/biopsy.csv')
    pcol = 4
    G = gini(df.ix[:, pcol])
    start = time.clock()
    gini_split(G, df, [2, 0], pcol)
    print(time.clock() - start)












def mse(x,y):
    mean_y = y.mean()
    return 1/len(x)*sum((x-mean_y)**2)

def mse_split(df,column_ids,predict):

    mse_values = []
    split_values = []

    for feature in column_ids:
        x = df.ix[:,feature] #a single feature column x in the dataframe 
        cat_x = df.ix[:,feature].astype('category')
        cats = cat_x.cat.categories #get the possible categories in x
        cats = cats[1:]
        all_splits = []
        for value in cats: #do a split for every category in x
            df_l = df[x<value]
            df_r = df[x>=value]
            #mse
            temp_mse = mse(df_l.ix[:,feature],df_l.ix[:,predict])+mse(df_r.ix[:,feature],df_l.ix[:,predict])
            all_splits.append(temp_mse)

        min_gini_value = min(all_splits)
        min_index = all_splits.index(min_gini_value)
        best_split = cats[min_index] #best split based on min gini impurity

        mse_values.append(min_gini_value)
        split_values.append(best_split)

    global_min_gini = min(mse_values) #best gini of all features
    best_index = mse_values.index(global_min_gini)
    best_feature = column_ids[best_index]
    best_split = split_values[best_index]
    df_l = df[df.ix[:,best_feature]<best_split]
    df_r = df[df.ix[:,best_feature]>=best_split]

    return best_feature,best_split, df_l,df_r
