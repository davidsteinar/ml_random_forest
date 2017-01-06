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

args = parser.parse_args()

# make data set
df = pd.read_csv(args.dataname)

bag = bagging(df, args.numtrees)

forrest = Forrest()
oob_errors []

# make the trees, for each iter calc oob error
for i in range(len(bag)):
    forrest.add(bag[i][0], bag[i][1])
    oob_errors.append(forrest.oob())


# plot num trees against oob error
num_trees = np.asarange(len(bag))
plt.xlabel('Number of trees')
plt.ylabel('OOB error')
plt.plot(num_trees, oob_errors)

plt.figure()


plt.show()
