# Author Hesam Pakdaman
from dtree import *

class Forrest():

    def __init__(self, df, pcol, isreg, numfeat):
        self.isreg = isreg
        self.numfeat = numfeat
        self.orgdf = df
        self.pcol = pcol
        self.trees = []

    def add_tree(self, data):
        df = data[0]                # bootstrap df
        oob_indx = data[1]          # oob indices

        tree = Dtree(oob_indx, self.pcol, self.isreg, self.numfeat)
        tree.grow(df)

        self.trees.append(tree)     # append to list of trees

    def oob_error(self):
        toterr = 0
        for tree in self.trees:
            toterr += tree.error(self.orgdf)
        return toterr
