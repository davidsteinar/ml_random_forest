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


# GROWING FOREST


forest = Forest(orgdf, 1, args)

print('Growing Forest with %s trees' %(args.numtrees))
start = time.clock()

for i in range(args.numtrees):
    forest.add_tree(bag[i])

ref_error = forest.error_OOB()

fin = 'Time to grow %d trees with %d features: %.2fs '\
            %(forest.numtrees, args.feat, time.clock() - start)

print(fin)




#-----------------------------------------------------------------------




increased_err = []                      # incr error due to leftout feat
leftout_feat = []                       # which feature left out


forest.trees[0].print_tree()


print(orgdf.shape)
# loop over features and skip predictive column
for i in range(orgdf.shape[1]-1):
    if(i != args.pcol):
        df = permute(orgdf, i)
        forest.orgdf = df
        error = forest.error_OOB()
        increased_err.append((error - ref_error) / ref_error)
        leftout_feat.append(i + 1)

print(increased_err)
plt.scatter(leftout_feat, increased_err)
plt.show()
