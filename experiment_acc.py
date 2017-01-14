# Author Hesam Pakdaman


from bagging import *
from forest import *
from importance import *
from params import *
from process_data import *
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time




#-----------------------------------------------------------------------


# PARSER


parser = argparse.ArgumentParser()

set_arg = parser.add_argument_group('Settings')
set_arg.add_argument('--file', type=str, default='a')
set_arg.add_argument('--maxfeat', type=int, default=3)
set_arg.add_argument('--feat', type=int, default=1)
set_arg.add_argument('--numtrees', type=int, default=100)

parser = parser.parse_args()









#-----------------------------------------------------------------------



# MAKE DATASETS, BAGGING


datasets = ['glass', 'breast_cancer', 'diabetes', 'sonar']

for s in datasets:
    parser.file = s
    args = Params(parser)

    orgdf = pd.read_csv(args.dataname)
    orgdf, testdf = process_data(orgdf, args)

    names = orgdf.columns.values           # names of features in data
    assert(args.maxfeat < orgdf.shape[1])  # do not exceed max cols

    size = orgdf.shape[0]                  # size for bootstrap
    bag = bagging(orgdf, args.numtrees, size)



#-----------------------------------------------------------------------


    # GROWING FOREST

    M = orgdf.shape[1] - 1
    F = int(math.log2(M) + 1)

    forest_1 = Forest(orgdf, 1, args)
    forest_2 = Forest(orgdf, F, args)

    s_forest_1 = Forest(orgdf, 1, args)
    s_forest_2 = Forest(orgdf, F, args)

    s_forest_1.add_tree([orgdf, 0])
    s_forest_2.add_tree([orgdf, 0])


    # start = time.clock()
    for i in range(args.numtrees):
        forest_1.add_tree(bag[i])
        forest_2.add_tree(bag[i])
    # print(time.clock() - start)


    e1 = forest_1.error(testdf) * 100
    e2 = forest_2.error(testdf) * 100

    fin = 'Data set: %s \n F1: \t %.3f \n FM: \t %.3f \n' %(s, e1, e2)
    print(fin)


    e1 = s_forest_1.error(testdf) * 100
    e2 = s_forest_2.error(testdf) * 100

    fin = 'One tree \n F1: \t %.3f \n FM: \t %.3f \n' %(e1, e2)
    print(fin)




#-----------------------------------------------------------------------

