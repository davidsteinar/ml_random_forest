# Author Hesam Pakdaman
from tree import *

class Forrest():

    def __init__(self, df, predcol, isRegression):
        self.trees = []
        self.reg = isRegression
        self.pcol = predcol
        self.df = df

    def add_tree(self, data):
        df = data[0]
        oob_indx = data[1]

        tree = Dtree(self, oob_indx, predcol, is_reg)
        tree.grow(df)
        self.trees.append(tree)

    def oob_error(self):
        toterr = 0
        for tree in self.trees():
            toterr += tree.error(self.df)
        return toterr
