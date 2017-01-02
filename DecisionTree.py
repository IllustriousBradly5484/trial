
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')
from sklearn import tree


# In[ ]:




# In[2]:

df1=pd.read_csv("C:\\Users\\Administrator\\Downloads\\train.csv",header=0);


# In[3]:

df1.head()


# In[4]:

cols=df1.shape


# In[5]:

cols


# In[6]:

cols=df1.shape[1]


# In[10]:

X=df1.iloc[:,1:cols]
X.head()


# In[11]:

Y=df1.iloc[:,0]
Y.head()


# In[8]:

clf = tree.DecisionTreeClassifier()


# In[32]:

clf


# In[12]:

clf = clf.fit(X, Y)


# In[13]:

with open("titanic.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)


# In[14]:

import pydotplus
import io


# In[47]:




# In[57]:




# In[15]:

from IPython.display import Image  


# In[77]:

feature_names=['pclass','Sex','Age','SibSp','parch','fare','embarked']
target_names=['servived']
dot_data = tree.export_graphviz(clf, out_file='abc', 
                         feature_names=feature_names,  
                         class_names=target_names,  
                         filled=True, rounded=True)                          
  


# In[16]:

dot_data = tree.export_graphviz(clf, out_file='abc')
dot_data


# In[17]:

graph = pydotplus.graph_from_dot_file('titanic.dot')


# In[18]:

from IPython.display import Image 


# In[19]:

graph.write_pdf("titanic.pdf")


# # this worked

# In[ ]:



