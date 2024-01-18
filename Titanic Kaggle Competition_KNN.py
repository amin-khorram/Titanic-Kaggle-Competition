#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import sklearn


# In[3]:


os.chdir("G:\\Tech Job Hunters Youtube Channel\\Dataset\\Titanic Tutorial")
train_path = "G:\\Tech Job Hunters Youtube Channel\\Dataset\\Titanic Tutorial\\train.csv"
test_path = "G:\\Tech Job Hunters Youtube Channel\\Dataset\\Titanic Tutorial\\test.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


# In[4]:


train_data.head(10)


# In[5]:


test_data.head(10)


# In[7]:


from sklearn.neighbors import KNeighborsClassifier as knn


# In[9]:


y = train_data['Survived']
features = ["Pclass", "Sex","SibSp","Parch"]
x = pd.get_dummies(train_data[features])
x_test = pd.get_dummies(test_data[features])


# In[10]:


model = knn(n_neighbors = 3)


# In[11]:


model = model.fit(x,y)


# In[15]:


predictions = model.predict(x_test)


# In[18]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv("submission_KNN_1.csv", index=False)

