class Params():

    def __init__(self, args):
        self.frac = 0.9
        self.isreg = False
        self.numtrees = args.numtrees
        self.feat = args.feat

        if(args.file == 'breast_cancer'):
            self.dataname = './datasets/breast_cancer.csv'
            self.conv = False
            self.maxfeat = args.maxfeat
            self.pcol = 9

        elif(args.file == 'sonar'):
            self.dataname = './datasets/sonar.csv'
            self.conv = False
            self.maxfeat = args.maxfeat
            self.pcol = 60

        elif(args.file == 'iris'):
            self.dataname = './datasets/iris.csv'
            self.conv = True
            self.maxfeat = args.maxfeat
            self.pcol = 4

        elif(args.file == 'biopsy'):
            self.dataname = './datasets/biopsy.csv'
            self.conv = False
            self.maxfeat = args.maxfeat
            self.pcol = 9

        elif(args.file == 'ecoli'):
            self.dataname = './datasets/ecoli.csv'
            self.conv = False
            self.maxfeat = args.maxfeat
            self.pcol = 7

        elif(args.file == 'diabetes'):
            self.dataname = './datasets/diabetes.csv'
            self.conv = False
            self.maxfeat = args.maxfeat
            self.pcol = 8

        else:
            print('No such file')
