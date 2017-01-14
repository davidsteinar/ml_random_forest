# Author Hesam Pakdaman

from gini_split import *
from randomFeature import *
import graphviz as gv
import time


class Dtree():


#-----------------------------------------------------------------------



# INNER CLASS OF NODES

    class Node():
        def __init__(self, feature, split):
            self.feature  = feature
            self.split    = split
            self.lchild   = None
            self.rchild   = None
            self.mode     = None
            self.isleaf   = False
            self.featname = None



#-----------------------------------------------------------------------



# CONSTRUCTOR FOR DECISION TREE

    def __init__(self, oob_indx, F, classes, args):
        self.root = None
        self.pcol = args.pcol
        self.isreg = args.isreg
        self.oob_indx = oob_indx
        self.numfeat = F
        self.nodes = 0
        self.timer = 0
        self.classes = classes




#-----------------------------------------------------------------------



# GROWING THE TREE


    def split_data(self, G,  df):
        # best feature and its split, check if for regression
        if (self.isreg):
            return None
        else:
            rnd_features = rand_indx(df, self.pcol, self.numfeat)
            feature, split, df_left, df_right, G1, G2 =\
                                    gini_split(G, df, rnd_features, self.pcol, self.classes)
        return feature, split, df_left, df_right, G1, G2


    def grow_recursion(self, G, df):
        self.nodes += 1

        start = time.clock()

        feature, split, df_left, df_right, G1, G2 =\
                        self.split_data(G, df)

        self.timer += time.clock() - start

        node = Dtree.Node(feature, split)
        node.featname = df.columns[feature]

        # if the node is not a leaf
        if(feature != -1):
            if(not df_left.empty):
                node.lchild = self.grow_recursion(G1, df_left)
            if(not df_right.empty):
                node.rchild = self.grow_recursion(G2, df_right)

        # assign value if the node is a leaf
        else:
            node.mode = self.assign_mode(df)
            node.isleaf = True
        return node


    def grow(self, df):
        # start = time.clock()

        G = gini(df.iloc[:, self.pcol], self.classes)
        self.root = self.grow_recursion(G, df)

        # end = time.clock() - start
        # print(self.timer)

    def assign_mode(self, df):
        if (self.isreg):
            return df.iloc[:, self.pcol].mean(axis=0)
        else:
            return df.iloc[:, self.pcol].value_counts().idxmax()




#-----------------------------------------------------------------------



# PREDICTION


    def predict_recursion(self, node, x):
        if(node.isleaf):
            return node.mode
        else:
            if(x.iloc[node.feature] >= node.split):
                return self.predict_recursion(node.rchild, x)
            else:
                return self.predict_recursion(node.lchild, x)


    def predict(self, x):
        return self.predict_recursion(self.root, x)



#-----------------------------------------------------------------------




# OOB
    def is_oob(self, i):
        return i in self.oob_indx





#-----------------------------------------------------------------------



# VISUALIZATION

    def make_graph(self, node):
        parentstr = '%s \n <= %s' %(node.featname, str(node.split))

        if(node.lchild is not None):
            if(node.lchild.isleaf):
                lchildstr = str(node.lchild.mode)
                self.G.node(str(node.lchild), lchildstr)
            else:
                lchildstr = '%s \n <= %s' %(node.lchild.featname,\
                        str(node.lchild.split))
                self.G.node(str(node.lchild), lchildstr)
                self.make_graph(node.lchild)

            self.G.edge(str(node), str(node.lchild))

        if(node.rchild is not None):
            if(node.rchild.isleaf):
                rchildstr = str(node.rchild.mode)
                self.G.node(str(node.rchild), rchildstr)
            else:
                rchildstr = '%s \n <= %s' %(node.rchild.featname,\
                        str(node.rchild.split))
                self.G.node(str(node.rchild), rchildstr)
                self.make_graph(node.rchild)

            self.G.edge(str(node), str(node.rchild))

    def print_tree(self):
        self.G = gv.Graph(format='svg')
        parentstr = '%s \n <= %s' %(self.root.featname,\
                str(self.root.split))
        self.G.node(str(self.root), parentstr)
        self.make_graph(self.root)
        self.G.render(filename='./img/'+str(self))



#-----------------------------------------------------------------------




def testTree():
    df = pd.read_csv('./datasets/iris.csv')
    tree = Dtree([0], 4, False, 2)
    tree.grow(df)

    x = pd.Series(np.random.normal(0, 1, size=df.shape[1]))
    print(tree.predict(x))
    tree.print_tree()



