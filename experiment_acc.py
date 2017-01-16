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
set_arg.add_argument('--feat', type=int, default=2)
set_arg.add_argument('--numtrees', type=int, default=100)

parser = parser.parse_args()









#-----------------------------------------------------------------------



# MAKE DATASETS, BAGGING


# datasets = ['breast_cancer', 'ecoli', 'diabetes', 'sonar']
datasets = ['sonar']


errors1 = []
errors2 = []

s_errors1 = []
s_errors2 = []


reps = 10
for s in datasets:
    for i in range(reps):
        parser.file = s
        args = Params(parser)

        orgdf = pd.read_csv(args.dataname)
        orgdf = orgdf.iloc[np.random.permutation(len(orgdf))]
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

        print(e1, e2)

        errors1.append(e1)
        errors2.append(e2)


        e1_s = s_forest_1.error(testdf) * 100
        e2_s = s_forest_2.error(testdf) * 100

        s_errors1.append(e1_s)
        s_errors2.append(e2_s)



    e1 = sum(errors1) / reps
    e2 = sum(errors2) / reps



    fin = 'Data set: %s \n F1: \t %.3f \n FM: \t %.3f \n' %(s, e1, e2)
    print(fin)


    e1 = sum(s_errors1) / reps
    e2 = sum(s_errors2) / reps

    fin = 'One tree \n F1: \t %.3f \n FM: \t %.3f \n' %(e1, e2)
    print(fin)





    #-----------------------------------------------------------------------

