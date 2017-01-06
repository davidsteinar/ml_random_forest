#======= Pantelis ========#

import pandas as pd
import numpy as np
from random import shuffle

def gini_index(d_class):
    
    count = 0
    prob = 0
    total = 0
    
    d_class.cat.codes.align
  
    
    for i in range(d_class.cat.codes[len(d_class)-1]+1):
        print("class = ", i)
        for j in range(len(d_class)-1):
            if  d_class.cat.codes[j] == i:
                count += 1
        print("count = ", count)        
        prob = (count / (len(d_class)-1))**2
        total += prob
        print(total)
        count = 0
        
    
    
    
    
    g = 1 - total
    return g

df = pd.read_csv('biopsy.csv')
d_class = df.iloc[:,-1].astype("category")
gini = gini_index(d_class)
print("gini index =",gini)
