
# author Pantelis

import pandas as pd
import numpy as np
from random import *


def TT_sets(df, p):

    train_set = df.sample(frac = p)
    test_set = df.drop(train_set.index[:])  
    return train_set, test_set

       

def combination(df, L, pcol, state):  # L: number of column to combine, pcol: the column which include the classes
    
    idx = []
    mu = []
    std = []    
    comb_col = 0
    
    
    while len(idx) < L:
        num = np.random.randint(0,df.shape[1])
        
        if num not in idx and num != pcol:
            idx.append(num)
    #print(idx)    
            
            
                  
    w = np.random.uniform(-1,1,L)
    for i in range(0,len(idx)):
        
        if state == True:
                m = np.mean(train_set.iloc[:,idx[i]])
                mu = np.append(mu,m)
                s = np.std(train_set.iloc[:,idx[i]])
                std = np.append(std,s)
                df.iloc[:,idx[i]] = (df.iloc[:,idx[i]] - mu[i])/std[i]
                
        #print(df.iloc[:,idx[i]])
        comb_col += df.iloc[:,idx[i]]*w[i]
        
        
    return comb_col                  #   ,idx,w uncomment for test_comb function
    


def df_extension(extended_df,comb_col, F):    # the number of the features we choose after 
    
    
    #for col in extended_df.columns:
    if 'combination'not in extended_df.columns:
        extended_df['combination'] = comb_col[:]
    else:
        print("Combination col is already there")
            #extended_df.to_csv('iris.csv')
    
    
    return extended_df  
    

    
    
    
    
    
def test_comb(df, idx, w):
    
    for i in range(0,L):
        
        print("---------> %d Column" %(idx[i]),"\n", "\n",w[i] * df.iloc[:,idx[i]], "\n" )
    print("Weights: ", "\n","\n", w)
    print("Combination =", "\n", "\n",comb_col, "\n","\n")
    

    
################################################# ignore it
    
# For deleting columns in df     

#for col in df.columns:
    #if 'combination' in col:
         #df.drop(col, axis=1, inplace=True)
#print(df)

#################################################
  


        
        

filename = 'iris.csv'
df = pd.read_csv(filename)
extended_df = pd.read_csv(filename)
L = 3
F = 2
p = 0.8            # persentage of training sets
state = False      # change to true if input vars are incommensurable
pcol = 4           # class column change manually


train_set, test_set = TT_sets(df, p)
combined_col = combination(df, L, pcol, state)

extended_df = df_extension(extended_df,combined_col, F)
#print(extended_df)





