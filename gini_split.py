# Author: David Steinar

import pandas as pd
import time

def gini(f):
    f = f.astype('category')
    categories = f.cat.categories
    sum = 0
    for category in categories:
        p = f[f == category].count()/f.count()
        sum += p**2
    return 1 - sum

def gini_split(df,column_ids,predict):

    gini_values = []
    split_values = []


    for feature in column_ids:

        x = df.ix[:,feature] #a single feature column x in the dataframe
        cat_x = df.ix[:,feature].astype('category')
        cats = cat_x.cat.categories #get the possible categories in x

        all_splits = []



        for value in cats: #do a split for every category in x
            df_l = df[x<value]
            df_r = df[x>=value]
            N1 = len(df_l)
            N2 = len(df_r)
            N = N1+N2
            #calculate weighted gini for the split
            s_gini = N1/N*gini(df_l.ix[:,predict])+N2/N*gini(df_r.ix[:,predict])
            all_splits.append(s_gini)




        min_gini_value = min(all_splits)
        min_index = all_splits.index(min_gini_value)
        best_split = cats[min_index] #best split based on min gini impurity

        gini_values.append(min_gini_value)
        split_values.append(best_split)




    global_min_gini = min(gini_values) #best gini of all features
    best_index = gini_values.index(global_min_gini)
    best_feature = column_ids[best_index]
    best_split = split_values[best_index]
    df_l = df[df.ix[:,best_feature]<best_split]
    df_r = df[df.ix[:,best_feature]>=best_split]

    if sum(gini_values) < 0.001:
        best_feature = -1

    return best_feature,best_split, df_l,df_r

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
