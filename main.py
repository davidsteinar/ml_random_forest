# Author Hesam Pakdaman

from bagging import *
from forrest import *
import argparse
import numpy as np
import pandas as pd
import time

# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
set_arg = parser.add_argument_group('Settings')
set_arg.add_argument('--conv', type=bool, default=True)
set_arg.add_argument('--dataname', type=str, default='./datasets/iris.csv')
set_arg.add_argument('--isreg', type=bool, default=False)
set_arg.add_argument('--numfeat', type=int, default=2)
set_arg.add_argument('--numtrees', type=int, default=500)
set_arg.add_argument('--pcol', type=int)
args = parser.parse_args()

# make data set
df = pd.read_csv(args.dataname)

# cat names to int
if(not args.isreg and args.conv):
    df.iloc[:, args.pcol] = df.iloc[:, args.pcol].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

size = 2 * df.shape[0] // 3             # use 2/3 of set
bag = bagging(df, args.numtrees, size)

# grow forrest
forrest = Forrest(df, args.pcol, args.isreg, args.numfeat)

# make the trees, for each iter calc oob error
oob_errors = []
for i in range(len(bag)):
    forrest.add_tree(bag[i])
    # oob_errors.append(forrest.oob_error())


print(len(forrest.trees))               # num of trees

forrest.trees[0].print_tree()
print(forrest.trees[0].nodes)

# plot num trees against oob error
# num_trees = np.arange(len(bag))
# plt.xlabel('Number of trees')
# plt.ylabel('OOB error')
# plt.plot(num_trees, oob_errors)

# plt.figure()
# plt.show()
