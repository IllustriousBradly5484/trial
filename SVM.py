
# coding: utf-8

# In[15]:

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
get_ipython().magic(u'matplotlib inline')


# In[3]:

df = pd.read_txt('C:\Users\Priya\Documents\Blood_donation.txt')


# In[5]:

df = pd.read_txt('C:\Users\Priya\Documents\Blood_donation.txt')


# In[6]:

f = open("Blood_donation.txt")
print(f.read())


# In[17]:

df = pd.read_csv("C:\Users\Priya\Documents\Blood_donation.txt",sep=",",names=['Recency (months)','Frequency (times)','Monetary (c.c. blood)','Time (months)','whether he/she donated blood in March 2007])


# In[18]:

df.head()


# In[16]:

data = pd.DataFrame(df['X'], columns=['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)', 'Time (months)', 'whether he/she donated blood in March 2007' ])  
data['y'] = df['y']

positive = data[data['y'].isin([1])]  
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(positive['Recency (months)'], positive['Frequency (times)'],positive['Monetary (c.c. blood)'],positive['Time (months)'],positive['whether he/she donated blood in March 2007'], s=50, marker='x', label='Positive')  
ax.scatter(negative['Recency (months)'], negative['Frequency (times)'],negative['Monetary (c.c. blood)'],negative['Time (months)'],negative['whether he/she donated blood in March 2007'], s=50, marker='o', label='Negative')  
ax.legend() 


# In[22]:

X=df.iloc[:,0:4]


# In[25]:

Y=df.iloc[:,4]


# In[23]:

X.head()


# In[26]:

Y.head()


# In[29]:

data = pd.DataFrame(df['Recency (months)'], columns=['Recency (months)', 'Frequency (times)', 'Monetary (c.c. blood)', 'Time (months)', 'whether he/she donated blood in March 2007' ])  
data['y'] = df['whether he/she donated blood in March 2007']

positive = data[data['y'].isin([1])]  
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(positive['Recency (months)'], positive['Frequency (times)'],positive['Monetary (c.c. blood)'],positive['Time (months)'],positive['whether he/she donated blood in March 2007'], s=50, marker='x', label='Positive')  
ax.scatter(negative['Recency (months)'], negative['Frequency (times)'],negative['Monetary (c.c. blood)'],negative['Time (months)'],negative['whether he/she donated blood in March 2007'], s=50, marker='o', label='Negative')  
ax.legend() 


# In[32]:

positive = df[df['whether he/she donated blood in March 2007'].isin([1])]  
negative = df[df['whether he/she donated blood in March 2007'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(positive['Recency (months)'], positive['Frequency (times)'], s=50, c='b', marker='o', label='Donating Blood')  
ax.scatter(negative['Recency (months)'], negative['Frequency (times)'], s=50, c='r', marker='x', label='Not Donating Blood')  
ax.legend()  
ax.set_xlabel('Exam 1 Score')  
ax.set_ylabel('Exam 2 Score') 


# In[42]:

from sklearn import svm  
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)  
svc 


# In[35]:

svc.fit(df[['Recency (months)', 'Frequency (times)']], data['y'])  
svc.score(df[['Recency (months)', 'Frequency (times)']], data['y'])


# In[37]:

svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)  
svc2.fit(df[['Recency (months)', 'Frequency (times)']], data['y'])  
svc2.score(df[['Recency (months)', 'Frequency (times)']], data['y'])


# In[51]:

data['SVM 1 Confidence'] = svc.decision_function(df[['Recency (months)', 'Frequency (times)']])

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(df['Recency (months)'], df['Frequency (times)'], s=50, c=data['SVM 1 Confidence'], cmap='seismic')  
ax.set_title('SVM (C=1) Decision Confidence')  


# In[39]:

data['SVM 2 Confidence'] = svc2.decision_function(df[['Recency (months)', 'Frequency (times)']])

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(df['Recency (months)'], df['Frequency (times)'], s=50, c=data['SVM 2 Confidence'], cmap='seismic')  
ax.set_title('SVM (C=100) Decision Confidence')  


# In[40]:

def gaussian_kernel(x1, x2, sigma):  
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))

x1 = np.array([1.0, 2.0, 1.0])  
x2 = np.array([0.0, 4.0, -1.0])  
sigma = 2  
gaussian_kernel(x1, x2, sigma) 


# In[43]:

data1=svc.fit(X,Y)  
#svc.score(df[['Recency (months)', 'Frequency (times)']], data['y'])


# In[46]:

plt.plot(df)


# In[47]:

svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)  
svc2.fit(X,Y)  
svc2.score(X,Y)


# In[49]:

data['SVM 1 Confidence'] = svc.decision_function(X)

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(X, s=50, c=data['SVM 1 Confidence'], cmap='seismic')  
ax.set_title('SVM (C=1) Decision Confidence') 


# In[50]:

data['SVM 1 Confidence'] = svc.decision_function(df[['Recency (months)', 'Frequency (times)']])

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(df['Recency (months)'], df['Frequency (times)'], s=50, c=data['SVM 1 Confidence'], cmap='seismic')  
ax.set_title('SVM (C=1) Decision Confidence')  


# In[ ]:



