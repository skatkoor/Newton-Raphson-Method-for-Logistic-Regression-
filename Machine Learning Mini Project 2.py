#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re
from sklearn import metrics 
from sklearn.metrics import accuracy_score


# Giving names to the rows

# In[2]:


df_col = ['id','Clump_Thickness','Uniformity_of_Cell_Size','Uniformity_of_Cell_Shape','Marginal_Adhesion','Single_Epithelial_Cell_Size','1.3','Bland_Chromatin','Normal_Nucleoli','Mitoses','Cancer_Type']


# Importing data

# In[3]:


df = pd.read_csv(r'C:\Users\Sumanth\OneDrive\UAB\Fall 2022\Machine Learning\Newton Raphson\breast_cancer_wisconsin.data',encoding='latin-1',names=df_col)
df.head(3)


# 2 for benign, 4 for malignant

# In[4]:


df.shape


# In[5]:


df.corr()


# From the correlation we get to know that the row id has nothing to do with the end result and can be removed

# In[6]:


df['1.3'].head(3)


# Removing the columns with '?'

# In[7]:


df1 = df.drop('id',axis = 1)
df2 = df1[df1['1.3'] == '?']
df3 = df1.drop([23,40,139,145,158,164,235,249,275,292,294,297,315,321,411,617])
df3['1.3'].unique()


# In[8]:


df3.dtypes


# In[9]:


df2.head()


# In[10]:


df3['1.3'].unique()


# Converting the datatype from object to float

# In[11]:


df5 = df3.replace('[^\d.]','',regex = True).astype(float)
df5.head()


# In[12]:


df5['Cancer_Type'] = df5['Cancer_Type'].replace(2,0)
df5['Cancer_Type'] = df5['Cancer_Type'].replace(4,1)


# In[13]:


df5.dtypes


# In[14]:


df5.isnull().sum()


# ![Screenshot%202022-11-06%20154504-2.png](attachment:Screenshot%202022-11-06%20154504-2.png)

# In[15]:


train_set = 0.8 
number_of_split=10
Lr_overall = 0 
Sk_overall = 0 
randomnumbers = np.random.rand(len(df5)) < train_set
train_df = df5[randomnumbers]
test_df = df5[~randomnumbers]
train_x = train_df.drop(["Cancer_Type"],axis=1)  
train_y = train_df["Cancer_Type"].values
test_x = test_df.drop(["Cancer_Type"],axis=1)
test_y = test_df["Cancer_Type"].values
ap = np.array(test_x).shape[0]
Intercept = np.ones((ap, 1))


# In[16]:


def prediction (b, X):
        X = np.c_[X, Intercept]
        z = np.dot(b, X.transpose()) 
        probability = 1/(1 + np.exp(-z))
        predicted_result = [0 if i <0.5 else 1 for i in probability]
        
        return predicted_result


# In[17]:


def logistic_regr(X,Y):
    Iteration_overall=150
    Iteration_count=0
    conver_tol=0.000000001
    final=np.zeros(X.shape[1]+1)
    final1=[]
    Intercept_X=np.ones((X.shape[0],1))
    X=np.c_[X, Intercept_X]
    while((Iteration_count<=Iteration_overall)):
        Iteration_count+=1
        final1=final
        z = np.dot(final, X.transpose()) 
        probability = 1/(1 + np.exp(-z))
        grad_descent = np.dot((Y - probability),X)
        we = probability*(1-probability)
        xP = np.array([i*j for (i,j) in zip(X,we)])
        hesian = -1*np.dot(xP.transpose(), X)
        final = final - (np.dot(np.linalg.inv(hesian),grad_descent))
        fin = np.linalg.norm(final-final1)
        if(fin<=conver_tol):
            break
    return final


# In[18]:


for i in range(number_of_split):
    final=logistic_regr(np.array(train_x), train_y)
    predicted= prediction(final,np.array(test_x))
    test_Y = test_y
    accuracy  = metrics.accuracy_score(test_Y, predicted)
    Lr_overall += accuracy
    logistic_regg_avg=Lr_overall/number_of_split 
    print(round(accuracy,2))


# In[19]:


X=df.drop(["Cancer_Type","1.3"],axis=1)
Y=df['Cancer_Type'].values
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2)


# In[20]:


for i in range(number_of_split):
    logregr= LogisticRegression(max_iter = 150)
    logregr.fit(train_x,train_y)
    Sk_overall += logregr.score(test_x,test_y)
    
logistic_Sk_avg=Sk_overall/number_of_split


# Using a solver can increase the efficiency of Sk learn Logistic Regression

# Accuracy of Newton method VS Sklearn method

# In[21]:


print('Newton method :',logistic_regg_avg)
print('Sklearn Logistic regression:',logistic_Sk_avg)


# In[ ]:





# In[ ]:




