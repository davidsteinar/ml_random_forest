# Author Hesam Pakdaman

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bagging import *

parser = argparse.ArgumentParser()
set_arg = parser.add_argument_group('Settings')
set_arg.add_argument('--dataname', type=str, default='iris.csv')
set_arg.add_argument('--numtrees', type=int, default=500)
set_arg.add_argument('--isreg', type=bool, default=False)
set_arg.add_argument('--pcol', type=int)

args = parser.parse_args()

# make data set
df = pd.read_csv(args.dataname)

bag = bagging(df, args.numtrees)

# grow the forrest
forrest = Forrest(df, args.pcol, args.isreg)

# make the trees, for each iter calc oob error
oob_errors []
for i in range(len(bag)):
    forrest.add_tree(bag[i])
    oob_errors.append(forrest.oob_error())


# plot num trees against oob error
num_trees = np.asarange(len(bag))
plt.xlabel('Number of trees')
plt.ylabel('OOB error')
plt.plot(num_trees, oob_errors)

plt.figure()


plt.show()
