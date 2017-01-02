
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import pandas as pd


# In[2]:

df1=pd.read_csv("D:\\DataScience\\Kmeans\\K_meansHW_TomMitchell_data2.csv",names=['X','Y'],header=0);
df1.head();


# In[3]:

X = df1.X
Y = df1.Y


# In[4]:

get_ipython().magic('matplotlib inline')
plt.plot(X,Y,"+")


# In[10]:

ou1=KMeans(n_clusters=3);
ou1.fit(df1);
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig)
#colormap=np.array(['red','blue','green'])
ax.scatter(X,Y,c=ou1.labels_.astype(np.float));
plt.title('K menas CLustering')
plt.show


# In[ ]:



