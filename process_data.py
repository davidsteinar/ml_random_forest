# Author Hesam Pakdaman

from TT_sets import *

def process_data(df, args):
    # category names to int
    if(not args.isreg and args.conv):
        df.iloc[:, args.pcol] = df.iloc[:, args.pcol].astype('category')
        cat_columns = df.select_dtypes(['category']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    # get test set with (1-frac) and with train set (frac)
    return TT_sets(df, args.frac)
