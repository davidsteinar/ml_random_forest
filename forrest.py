# Author Hesam Pakdaman
from dtree import *

class Forrest():

    def __init__(self, df, pcol, isreg, numfeat):
        self.isreg = isreg
        self.numfeat = numfeat
        self.orgdf = df             # original dataset
        self.N = df.shape[0]        # num of obs
        self.pcol = pcol            # predictive column
        self.trees = []
        self.numtrees = 0

        # num of classes
        self.classes = df.iloc[:, pcol].unique().shape[0] 

    def add_tree(self, data):
        self.numtrees += 1

        df = data[0]                # bootstrap df
        oob_indx = data[1]          # oob indices

        tree = Dtree(oob_indx, self.pcol, self.isreg, self.numfeat)
        tree.grow(df)

        self.trees.append(tree)     # append to list of trees

    def oob_error(self):
        error = 0
        num_oobs = 0

        for i in range(self.N):

            votes = np.zeros(self.classes)

            # calculate OOB error for obs i
            for j in range(self.numtrees):
                if(self.trees[j].is_oob(i)):

                    p_class = \
                        self.trees[j].predict(self.orgdf.iloc[i, :])

                    votes[p_class] += 1      # vote on predicted class

            # it can happen that none of the trees had oob
            # that is why we check that we at least get one vote
            if(sum(votes) != 0):
                num_oobs += 1

                # class that got majority votes and against true label
                if(votes.argmax() != self.orgdf.iloc[i, self.pcol]):
                    error += 1

        return error / num_oobs
