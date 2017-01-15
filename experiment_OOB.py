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
set_arg.add_argument('--file', type=str, default='sonar')
set_arg.add_argument('--maxfeat', type=int, default=8)
set_arg.add_argument('--numtrees', type=int, default=100)
set_arg.add_argument('--feat', type=int, default=2)

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


allinone_oob = np.zeros([10,args.maxfeat])
allinone_test = np.zeros([10,args.maxfeat])


# loop over the the num of features, including maxfeat
for t in range(10):
    df_train, df_test = process_data(df, args)
    for i in range(1, args.maxfeat):

        size = df.shape[0]          # use same size as data for bootstrap
        bag = bagging(df_train, args.numtrees, size)

	#print('Growing %s trees with %s features' %(args.numtrees, i))
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
        allinone_oob[t,i] = forest.error_OOB()
        allinone_test[t,i] = forest.error(df_test)

np.savetxt("sonar_oob.csv", allinone_oob, delimiter=",")
np.savetxt("sonar_test.csv", allinone_test, delimiter=",")

x = range(8)
oob_plot = np.mean(allinone_oob,axis=0)
test_plot = np.mean(allinone_test,axis=0)

plt.plot(x,oob_plot,label="oob")
plt.plot(x,test_plot,label="test")
plt.xlabel('Number of features')
plt.ylabel("Error percent")
plt.legend()
plt.savefig("sonar_oob_experiment.png")



