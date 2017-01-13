# Author Hesam Pakdaman
from dtree import *

class Forest():

    def __init__(self, df, F, args):
        self.isreg = args.isreg
        self.numfeat = F
        self.orgdf = df             # original dataset
        self.pcol = args.pcol       # predictive column
        self.trees = []
        self.numtrees = 0
        self.args = args

        # num of classes
        self.classes = df.iloc[:, self.pcol].unique().shape[0]




    def add_tree(self, data):
        self.numtrees += 1

        df = data[0]                # bootstrap df
        oob_indx = data[1]          # oob indices

        tree = Dtree(oob_indx, self.numfeat, self.args)
        tree.grow(df)

        self.trees.append(tree)     # append to list of trees



#-----------------------------------------------------------------------



# ERRORS


    def error_test(self, df):
        # for every row, calculate error
        error = 0

        for i in range(df.shape[0]):
            votes = np.zeros(self.classes)

            for j in range(self.numtrees):
                p_class = self.trees[j].predict(df.iloc[i, :])
                votes[p_class] += 1      # vote on predicted class

            # class that got majority votes against true label
            if(votes.argmax() != df.iloc[i, self.pcol]):
                error += 1

        return error / df.shape[0]




    def error_OOB(self):
        error = 0
        num_oobs = 0

        # for every obs in the original dataset
        for i in range(self.orgdf.shape[0]):

            votes = np.zeros(self.classes)

            # calculate OOB error for observation i
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



#-----------------------------------------------------------------------
