# Author Hesam Pakdaman


from bagging import *
from forest import *
from importance import *
from params import *
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
set_arg.add_argument('--file', type=str)
set_arg.add_argument('--maxfeat', type=int, default=3)
set_arg.add_argument('--feat', type=int, default=1)
set_arg.add_argument('--numtrees', type=int, default=100)

parser = parser.parse_args()
print(parser.file)
args = Params(parser)






#-----------------------------------------------------------------------



# MAKE ORG DATASET, BAGGING



orgdf = pd.read_csv(args.dataname)
# orgdf, _ = process_data(orgdf, args)
names = orgdf.columns.values           # names of features in data
assert(args.maxfeat < orgdf.shape[1])  # do not exceed max cols
size = orgdf.shape[0]                  # size for bootstrap
bag = bagging(orgdf, args.numtrees, size)



#-----------------------------------------------------------------------


# REF TREE




# reference forest and error
ref_forest = Forest(orgdf, 1, args)
print('Growing Reference Forest with %s trees' %(args.numtrees))
start = time.clock()

for i in range(args.numtrees):
    ref_forest.add_tree(bag[i])

fin = 'Time to grow %d trees with %d features: %.2fs '\
            %(ref_forest.numtrees, args.feat, time.clock() - start)
print(fin)

ref_err = ref_forest.error_OOB()
print(ref_err)





#-----------------------------------------------------------------------




perm_sets = permute(orgdf, args.pcol)                 # get all permute sets


increased_err = []                      # incr error due to leftout feat
leftout_feat = []                       # which feature left out


# loop over features and skip predictive column
for i in range(orgdf.shape[1]-1):

    print('Growing %s trees with feature %s permuted' %(args.numtrees, names[i]))
    bag = bagging(perm_sets[i], args.numtrees, size)

    forest = Forest(perm_sets[i], 1,  args)

    start = time.clock()

    # add tree to forest
    for j in range(args.numtrees):
        forest.add_tree(bag[j])

    fin = 'Time to grow %d trees with feature %s permuted: %.2fs '\
                %(forest.numtrees, names[i], time.clock() - start)
    print(fin)

    leftout_feat.append(i + 1)

    # increased error
    forest.orgdf = orgdf
    OOB_error = forest.error_OOB()
    print(OOB_error)
    increased_err.append((OOB_error - ref_err) / ref_err)





print(increased_err)
plt.scatter(leftout_feat, increased_err)
plt.show()
