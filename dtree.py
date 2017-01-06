# Author Hesam Pakdaman

class Dtree():

    def __init__(self):
        self.root = Node()

        class Node():
            def __init__(self):
                self.feature  = None
                self.split    = None
                self.lchild   = None
                self.rchild   = None
                self.mode     = None

    def make_child(df):

        feature, split = gini(df)

        node          = Node()
        node.feature  = feature
        node.split    = split

        if(feature > -1):
            # ask if feature is greater or equal to split value
            right_df = df[df.ix[:,feature] >= split] # true
            left_df = df[df.ix[:,feature] < split] # false

            node.lchild = make_child(df_left)
            node.rchild = make_child(df_right)
        else:
            self.mode = assign_mode(df)
        return node


    def create_tree(self, df):
        feature, split = gini(df)

        # ask if feature is greater or equal to split value
        right_df = df[df.ix[:,feature] >= split] #true
        left_df = df[df.ix[:,feature] < split] #false

        self.root.lchild = make_child(left_df)
        self.root.rchild = make_child(right_df)
