#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import math

data = pd.read_csv("train.csv")

remov = {}
remov["corr_filter"] = []
remov["nan_filter"] = []
remov["desc_filter"] = []


# In[7]:


''' filter: NaN values '''

def nan_filter(data):
    
    global remov
    
    nan_attr = {}
    sum_attr = data.isnull().sum()
        
    key = 0        
    for i in range(sum_attr.size):
        
        if(sum_attr.iloc[i] > 0):
            
            nan_attr[key] = [sum_attr.index[i], sum_attr.iloc[i]]
            key = key + 1
        
    for i in range(len(nan_attr)):
                
        if(nan_attr[i][1] >= 0.75 * data[nan_attr[i][0]].value_counts(dropna=False).sum()):
            data = data.drop(columns = nan_attr[i][0])
            remov["nan_filter"].append(nan_attr[i][0])
        elif(data[nan_attr[i][0]].dtype == 'object'):
            data[nan_attr[i][0]] = data[nan_attr[i][0]].fillna(data[nan_attr[i][0]].mode().iloc[0])
        else:
            data[nan_attr[i][0]] = data[nan_attr[i][0]].fillna(data[nan_attr[i][0]].mean())
    
    return data

data = nan_filter(data)
len(data.columns)
    


# In[8]:


''' filter: minimun correlation with target '''

def corr_filter(data, target):
    
    global remov
    
    cor = data.corr()
    cor_target = abs(cor[target])
    irrelevant_features = cor_target[cor_target < 0.6]
        
    data = data.drop(columns = irrelevant_features.index)
    remov["corr_filter"].append(irrelevant_features.index) 
    
    return data

data = corr_filter(data, "SalePrice")

len(data.columns)


# In[9]:


''' filter: desbalanced classes '''

def desc_filter(data):
    
    global remov
    
    objects = []
    classes_attr = {}
    
    for i in range(len(data.columns)):
    
        if(data[data.iloc[0].index[i]].dtype == "object"):
            objects.append(data.iloc[0].index[i]) 
        
    for i in range(len(objects)):
        
        class_size = data[objects[i]].value_counts().size
        classes_attr[i] = []                    

        for j in range(class_size):
            
            classes_attr[i].append(data[objects[i]].value_counts()[j])
            
    for i in range(len(classes_attr)):
        
            if(sum(classes_attr[i][1:]) < classes_attr[i][0]/2.5):
                
                remov["desc_filter"].append(objects[i])                              
                data = data.drop(columns = objects[i])
                      
    return data
                
            
data = desc_filter(data)
print(len(data.columns))


# In[10]:


print(remov['desc_filter'])


# In[148]:


data.head()


# In[ ]:




