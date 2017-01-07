# Author Hesam Pakdaman

class Dtree():

    def __init__(self, oob_indx, predcol, is_reg):
        self.root = None
        self.pcol = predcol
        self.isreg = is_reg
        self.isleaf = False
        self.oob_indx = oob_indx

        class Node():
            def __init__(self):
                self.feature  = None
                self.split    = None
                self.lchild   = None
                self.rchild   = None
                self.mode     = None

    def make_child(df):

        # best feature and its split, check if for regression
        if (self.isreg):
        else:
            rnd_features = randomIndicies(df, predict, m)
            feature, split, df_left, df_right =\
                    gini_split(df, rnd_features, self.pcol)

        node = Node(feature, split)

        # if the node is not a leaf
        if(feature > -1):
            node.lchild = make_child(df_left)
            node.rchild = make_child(df_right)

        # assign value if the node is a leaf
        else:
            self.mode = assign_mode(df)
            self.isleaf = True
        return node


    def grow(self, df):

        # best feature and its split, check if for regression
        if (self.isreg):
        else:
            rnd_features = randomIndicies(self.df, predict, m)
            feature, split, df_left, df_right =\
                    gini_split(self.df, rnd_features, self.pcol)

        self.root         = Node(feature, split)
        self.root.lchild  = make_child(left_df)
        self.root.rchild  = make_child(right_df)

    def rec_pred(node, x):
        if(node.is_leaf):
            return node.mode
        else:
            if(x.iloc(node.feature) >= node.split):
                rec_pred(node.rchild, x)
            else:
                rec_pred(node.lchild, x)

    def predict(self, x):
        return rec_pred(self.root, x)

    def error(self, df, predcol):
        toterr = 0
        oob_df = df.iloc[self.oob_indx]
        for i in range(len(oob_df)):
            pred_val = self.predict(df.iloc[i,:])
            if (pred_val != oob_df.iloc[i, predcol]):
                toterr += 1
        return toterr/len(oob_df)
