
# coding: utf-8

# In[3]:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import pandas as pd


# In[4]:

df1=pd.read_csv("D:\\DataScience\\Kmeans\\K_meansHW_TomMitchell_data1.csv",names=['X','Y'],header=0);
df1.head();


# In[5]:

df1


# In[9]:

centers = [[1, 1], [-1, -1], [1, -1]]
X = df1.X
Y = df1.Y


# In[12]:

get_ipython().magic('matplotlib inline')
plt.plot(X,Y,"+")


# In[52]:

ou1=KMeans(n_clusters=4);


# In[53]:

ou1.fit(df1);


# In[54]:

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[-4,-4, 4, 4])
colors = np.random.rand(3)
ax.scatter(X,Y,c=ou1.labels_.astype(np.float));
plt.show


# In[29]:

colors = np.random.rand(3)
colors


# In[51]:

ou1.labels_


# In[ ]:



