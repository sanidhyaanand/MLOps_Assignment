#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dvc.api
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier
import pickle
import json


# In[2]:


# reading dataset
with dvc.api.open("data/creditcard.csv", repo = "https://github.com/sanidhyaanand/MLOps_Assignment") as fd:
    df = pd.read_csv(fd)

#print(df.columns)

# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.4,random_state=42, stratify=Y) 

x = dvc.api.get_url(repo="https://github.com/sanidhyaanand/MLOps_Assignment", path="data/creditcard.csv")
print("metadata points to:" + x)


# In[12]:


# df.head()


# In[18]:


y = df.Class
X = df.drop(["Class", "Time"], axis = 1)


# In[40]:


#test-train split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("../data/processed/train.csv")
test_df.to_csv("../data/processed/test.csv")


# In[59]:


# training and prediction step

# clf = tree.DecisionTreeClassifier(criterion = "entropy")
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

clf = RandomForestClassifier(n_estimators=100, criterion = "entropy")
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[60]:


# saving model
filename = "../models/model.pkl"
pickle.dump(clf, open(filename, 'wb'))


# In[61]:


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Weighted F1 Score:", metrics.f1_score(y_test, y_pred, average = 'weighted'))

results = "Accuracy: " +  str(metrics.accuracy_score(y_test, y_pred)) + "\n" + "Weighted F1 Score:" + str(metrics.f1_score(y_test, y_pred, average = 'weighted'))


# In[62]:


with open('../metrics/acc_f1.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

