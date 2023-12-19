#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('C:/Users/makeai/Downloads/Applications_of_DL-main/Applications_of_DL-main/iris.csv', index_col=0)
df


# In[8]:


x = df.drop(columns='label',axis=1)
y = x['petal width (cm)']
x = x.drop(columns=['petal width (cm)'],axis=1)


# In[9]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[12]:


x


# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=423)


# In[19]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeRegressor

clf_dt = DecisionTreeRegressor()
clf_dt.fit(X_train, y_train)

pred_dt = clf_dt.predict(X_test)

print(clf_dt.score(X_train, y_train))


# In[20]:


mse = np.sqrt(mean_squared_error(pred_dt, y_test))
print('평균제곱근오차', mse)


# In[21]:


from sklearn.ensemble import RandomForestRegressor

rf_clf = RandomForestRegressor()
rf_clf.fit(X_train, y_train)

pred_rf = rf_clf.predict(X_test)

print(rf_clf.score(X_train, y_train))


# In[22]:


mse = np.sqrt(mean_squared_error(pred_rf, y_test))
print('평균제곱근오차', mse)


# In[23]:


from sklearn.linear_model import LinearRegression

clf_lr = LinearRegression()
clf_lr.fit(X_train, y_train)

pred_lr = clf_lr.predict(X_test)

print(clf_lr.score(X_train, y_train))


# In[24]:


mse = np.sqrt(mean_squared_error(pred_lr, y_test))
print('평균제곱근오차', mse)


# In[25]:


from sklearn.svm import SVR

clf_svm = SVR()
clf_svm.fit(X_train, y_train)

pred_svm = clf_svm.predict(X_test)

print(clf_svm.score(X_train, y_train))


# In[26]:


mse = np.sqrt(mean_squared_error(pred_svm, y_test))
print('평균제곱근오차', mse)


# In[ ]:




