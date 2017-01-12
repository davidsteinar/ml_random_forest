
# coding: utf-8

# In[171]:

# Pantelis

import pandas as pd

def TT_sets(df, p):

    train_set = df.sample(frac = p)
    test_set = df.drop(train_set.index[:])  
    return train_set, test_set


# In[ ]:




# In[172]:

len(df)


# In[173]:

df = pd.read_csv('biopsy.csv')
p = 0.8
x,y = sets(df, p)


# In[ ]:




# In[ ]:




# In[ ]:



