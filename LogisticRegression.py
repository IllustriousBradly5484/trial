
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')


# In[4]:

df1=pd.read_csv("D:\DataScience\Logistic Regression\ex2data1.csv",names=['Exam1','Exam2','Y'],header=0);
df1.head()


# In[7]:

cols = df1.shape[1] 
cols 


# In[8]:

X = df1.iloc[:,0:cols-1]  


# In[9]:

X.head()


# In[11]:

Y=df1.iloc[:,cols-1:cols]
Y.head()


# In[12]:

from sklearn import linear_model


# In[13]:

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)


# In[15]:

x_min, x_max = X['Exam1'].min() - .5, X['Exam1'].max() + .5


# In[16]:

y_min, y_max = X['Exam2'].min() - .5, X['Exam2'].max() + .5


# In[17]:

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


# In[18]:

Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])


# In[20]:

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))


# In[22]:

# Plot also the training points
plt.scatter(X['Exam1'], X['Exam2'], c=Y)
plt.xlabel('Exam Data')
plt.ylabel('Prediction')


# In[ ]:



