# Author Hesam Pakdaman

from bagging import *
from forest import *
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

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


size = df.shape[0]              # use same size as data for bootstrap
bag = bagging(df, args.numtrees, size)

# grow forest
forest = Forest(df, args.pcol, args.isreg, args.numfeat)

# make the trees, for each iter calc oob error
oob_errors = []
oob_trees = []
start = time.clock()
print('Starting to grow %s trees' %args.numtrees)

# calculate OOB error every k time
k = 5
for i in range(args.numtrees):
    forest.add_tree(bag[i])

    # calculate OOB error every k time
    if (i % k is 0 and i is not 0):
        print(str(i) + ' trees grown')
        oob_trees.append(i)
        oob_errors.append(forest.oob_error())

fin = 'Time to grow %d trees: %.2fs '\
            %(forest.numtrees, time.clock() - start)

print(fin)




# plot num trees against oob error
plt.figure()
plt.xlabel('Number of trees')
plt.ylabel('OOB error')
plt.plot(oob_trees, oob_errors)

plt.show()




'''
Testing only for iris data set
'''
def test_predict():

    forest.trees[0].print_tree()

    print(df.columns)
    x = pd.Series(np.asarray([5.1, 3.5, 1.4, 0.2]))
    print('Test point %s' %str(x.values))
    print(forest.trees[0].predict(x))

    x = pd.Series(np.asarray([4.3, 2.7, 0.4, 4.2]))
    print('Test point %s' %str(x.values))
    print(forest.trees[0].predict(x))
