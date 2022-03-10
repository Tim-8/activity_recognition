#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

phoneData = pd.read_csv('learning_data.csv')
phoneData


# In[36]:


phoneDF = pd.DataFrame(phoneData)


# In[37]:


phoneDF


# In[38]:


phoneDF.describe()


# In[39]:


class_train, class_test = train_test_split(phoneData, test_size=0.3)


# In[40]:


class_train


# In[41]:


class_test


# In[42]:


clf = MLPClassifier(hidden_layer_sizes=10, activation='relu',
                    solver='adam', max_iter=1000)


# In[43]:


clf.fit(class_train)


# In[44]:


phoneDF_x = phoneDF.drop("class", axis=1)


# In[45]:


phoneDF_y = phoneDF['class']


# In[46]:


phoneDF_x


# In[47]:


phoneDF_y


# In[48]:


data_train, data_test, target_train, target_test = train_test_split(
    phoneDF_x, phoneDF_y, test_size=0.3, random_state=0)


# In[49]:


clf = MLPClassifier(hidden_layer_sizes=10, activation='relu',
                    solver='adam', max_iter=1000)


# In[50]:


data_train


# In[51]:


clf.fit(data_train, target_train)


# In[52]:


clf.score(data_train, target_train)


# In[53]:


data_test


# In[54]:


clf.predict(data_test)


# In[55]:


target_test


# In[56]:


clf.loss_curve_


# In[57]:


plt.plot(clf.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()


# In[ ]:




