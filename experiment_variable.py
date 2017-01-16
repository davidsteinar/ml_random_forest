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
set_arg.add_argument('--numtrees', type=int, default=200)

parser = parser.parse_args()
print(parser.file)
args = Params(parser)









#-----------------------------------------------------------------------



# MAKE ORG DATASET, BAGGING



orgdf = pd.read_csv(args.dataname)
names = orgdf.columns.values           # names of features in data
assert(args.maxfeat < orgdf.shape[1])  # do not exceed max cols
size = orgdf.shape[0]                  # size for bootstrap
bag = bagging(orgdf, args.numtrees, size)



#-----------------------------------------------------------------------


# GROWING FOREST




f = Forest(orgdf, 2, args)

for i in range(args.numtrees):
    f.add_tree(bag[i])
    if(i % 20 == 0):
        print('%d trees grown' %(i+1))

ref_error = f.error_OOB()





#-----------------------------------------------------------------------




increased_err = []                      # incr error due to leftout feat
leftout_feat = []                       # which feature left out


print(orgdf.shape)
# loop over features and skip predictive column
for i in range(orgdf.shape[1]-1):
    if(i != args.pcol):
        error = 0
        df = permute(orgdf, i)
        f.orgdf = df
        error = f.error_OOB()
        increased_err.append((error - ref_error) / ref_error * 100)
        leftout_feat.append(i + 1)


print(increased_err)
plt.xlabel('variable')
plt.ylabel('increase precent')
plt.scatter(leftout_feat, increased_err)
plt.savefig('variable_importance.pdf')
plt.show()
