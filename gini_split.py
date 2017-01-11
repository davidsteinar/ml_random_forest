# Author: David Steinar, Hesam Pakdaman

import pandas as pd
import time

def gini(df):
    N = df.count()
    classes = df.unique()
    sum = 0
    for c in classes:
        p = df[df == c].count() / N
        sum += p**2
    return 1 - sum

def gini_split(G, df, column_ids, predict):

    if(G < 0.05 or df.shape[0] < 10):
        return -1, None, None, None, None, None

    gini_values = []
    split_values = []

    gini_vals = []
    gini_splits = []

    best_gini = None
    total_gini = 0

    for i in range(len(column_ids)):

        # get feature column
        x = df.iloc[:, column_ids[i]]
        sorted_x = x.sort_values()
        classes = len(x)

        k = 4

        # split into k-1 pieces
        for j in range(1, k):

            indx = j * classes // k
            split_value = sorted_x.iloc[indx]


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
                best_feature = column_ids[i]
                total_gini += G1 + G2

    df_l = df[df.iloc[:, best_feature] < best_split]
    df_r = df[df.iloc[:, best_feature] >= best_split]

    if (gini_gain <= 0):
        best_feature = -1


    return best_feature, best_split, df_l, df_r, G1, G2



def test_gini():
    df = pd.read_csv('./datasets/iris.csv')
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
