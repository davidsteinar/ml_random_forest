# Author Hesam Pakdaman


from bagging import *
from forest import *
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
set_arg.add_argument('--file', type=str, default='breast_cancer')
set_arg.add_argument('--maxfeat', type=int, default=3)

parser = parser.parse_args()
print(parser.file)
args = Params(parser)






#-----------------------------------------------------------------------



# MAKE DATASET, BAGGING


df = pd.read_csv(args.dataname)
assert(args.maxfeat < df.shape[1]) # do not exceed max cols





#-----------------------------------------------------------------------







# GROW THE FOREST

# grow forest with F random features

# for plotting error against numfeat
errors_oob = []
errors_test = []
num_features = []

# loop over the the num of features, including maxfeat
for i in range(1, args.maxfeat+1):

    df_train, df_test = process_data(df, args)
    size = df.shape[0]          # use same size as data for bootstrap
    bag = bagging(df_train, args.numtrees, size)

    print('Growing %s trees with %s features' %(args.numtrees, i))
    forest = Forest(df, i,  args)       # make tree with F features

    start = time.clock()

    # add args.numtrees to the forest
    for j in range(args.numtrees):
        forest.add_tree(bag[j])

    fin = 'Time to grow %d trees with %d features: %.2fs '\
                %(forest.numtrees, i, time.clock() - start)
    print(fin)

    # remember OOB/test errors with i num of features
    num_features.append(i)
    errors_oob.append(forest.error_OOB())
    errors_test.append(forest.error_test(df_test))




plt.plot(num_features, errors_oob, 'r')
plt.plot(num_features, errors_test, 'b')
# plt.axis([0, len(num_features), 0, 0.05])
plt.show()
