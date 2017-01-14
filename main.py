# Author Hesam Pakdaman

from bagging import *
from forest import *
from process_data import *
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time





#-----------------------------------------------------------------------

# PARSER



parser = argparse.ArgumentParser()

set_arg = parser.add_argument_group('Settings')
set_arg.add_argument('--conv', type=bool, default=False)
set_arg.add_argument('--dataname', type=str, default='./datasets/breast_cancer.csv')
set_arg.add_argument('--isreg', type=bool, default=False)
set_arg.add_argument('--numfeat', type=int, default=2)
set_arg.add_argument('--maxfeat', type=int, default=25)
set_arg.add_argument('--numtrees', type=int, default=100)
set_arg.add_argument('--pcol', type=int)
set_arg.add_argument('--frac', type=float, default=0.8)


args = parser.parse_args()




'''
Testing only
'''
def test_predict():

    # make data set
    df = pd.read_csv('./datasets/breast_cancer.csv')

    df_train, df_test = process_data(df, args)

    size = df.shape[0]              # use same size as data for bootstrap
    bag = bagging(df, args.numtrees, size)

    # grow forest with F random features
    forest = Forest(df, 2,  args)
    forest.add_tree(bag[0])
    forest.trees[0].print_tree()

    print(df.columns.values)

    x = pd.Series(np.asarray([4,1,1,3,2,1,3,1,1,0]))

    print('Test point %s' %str(x.values))
    print(forest.trees[0].predict(x))

    x = pd.Series(np.asarray([8,10,10,8,7,10,9,7,1,1]))
    print('Test point %s' %str(x.values))
    print(forest.trees[0].predict(x))

test_predict()
