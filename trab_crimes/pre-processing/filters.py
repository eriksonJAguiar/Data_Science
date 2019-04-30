#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from sklearn import preprocessing

remov = {}
remov["corr_filter"] = []
remov["nan_filter"] = []
remov["desc_filter"] = []

# In[2]:


''' filter: NaN values '''

def nan_filter(data):
    
    print(data.size)
    
    data = data.dropna()
    
    print(data.size)
    
    return data

def obj_filter(data):
    
    obj_data = data.select_dtypes(include=['object']).copy()

    col = list(obj_data.columns)

    for i in range(len(col)):
        
        #if(col[i] == 'Calendar' or col[i] == 'Hour'):
            #data[col[i]] = pd.to_numeric(obj_data[col[i]])
        #else:    
        obj_data[col[i]] = obj_data[col[i]].astype('category')
        data[col[i]] = obj_data[col[i]].cat.codes
    
    return data

def normalization(data):
    
    x = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    res = pd.DataFrame(x_scaled, columns= data.columns, index=data.index)
    
    return res

# In[3]:






