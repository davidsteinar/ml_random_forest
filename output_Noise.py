# author Pantelis

import pandas as pd
import numpy as np
import random

def output_Noise(df):
    d_class = df.iloc[:,-1].astype("category")
    d_class = d_class.cat.rename_categories([1,2])
    #d_class_list = d_class.tolist()
    new_d_class = d_class

    #print("Old d_class",d_class, "\n","\n","\n")
    n = (len(d_class)-1)%20              # number of values that will be changed 20% noise
    temp = []
    for i in range(1,n):

        while True:
            new_val = np.random.choice(range(1,len(d_class.cat.categories)+1))
            random_index = random.randint(0,len(d_class)-1)
            if new_d_class[random_index] != new_val:
                new_d_class[random_index] = new_val
                break

    #print("New d_class",new_d_class)    
    return(new_d_class)

    #temp.append(random_index)
    #if temp !=
