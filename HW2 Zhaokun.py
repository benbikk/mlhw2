#!/usr/bin/env python
# coding: utf-8

# # 1.Write a method for data preprocessing

# In the data preprocessing stage, I did not use the Age filling method Proferssor Hao provided in his example code. Professor Hao seperately fill the Nan values for Survived samples and Not Survived samples, namely use the information about label we want to classification in feature data preprocssing, which I believe will cause target leakage  since it would be using information that is only available after the event has occurred (i.e., the passenger survived). This could lead to overfitting and inflated performance estimates, since the model is effectively using future information to make predictions.Thus, I simply use the mean of age of all samples(i.e., no matter sample survives or not) to fill Age Nan values.

# In[81]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn import svm
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def getNumber(str):
    if str=="male":
        return 1
    else:
        return 2
    
def getS(str):
    if str=='S':
        return 1
    else:
        return 0
    
def getQ(str):
        if str=='Q':
            return 1
        else:
            return 0

def getC(str):
        if str=='C':
            return 1
        else:
            return 0


# In[45]:


def preprocess(train):
    del train['Name']
    del train['Cabin']
    del train['Fare']
    del train['Ticket']
    
    train["Gender"]=train["Sex"].apply(getNumber)
    del train['Sex']
    samplemean=train.Age.mean()
    
    train['age']=np.where(pd.isnull(train.Age), samplemean, train.Age)
   
    del train['Age']
    train.rename(columns={'age':'Age'}, inplace=True)
    train.dropna(inplace=True)
    
    train['S']=train['Embarked'].apply(getS)
    train['Q']=train['Embarked'].apply(getQ)
    train['C']=train['Embarked'].apply(getC)
    del train['Embarked']
    del train['PassengerId']
    return train


# # 2.Read and process the data for training

# In[46]:


train=pd.read_csv('train.csv',header = 0, dtype={'Age': np.float64})
train=preprocess(train)


# In[47]:


train.head()


# # 3.Read and process the data for testing

# In[48]:


test=pd.read_csv('test.csv',header = 0, dtype={'Age': np.float64})
test=preprocess(test)


# In[49]:


test.head()


# # 4.Models

# 4.1  Logistic Regression

# In[78]:


X = train[['Pclass', 'SibSp', 'Parch', 'Gender', 'Age', 'S', 'Q', 'C']]
 
y = train['Survived']

# create a logistic regression model
Logisticmodel = LogisticRegression(max_iter=1000, C=1.0)
y_pred = cross_val_predict(Logisticmodel, X, y, cv=5)

# Compute the performance metrics for the predicted target values and the true target values
cm = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)
auc = roc_auc_score(y, y_pred)

# Print out the performance metrics
print("Confusion matrix:")
print(cm)
print('\n')
print("AUC:", auc)
print('\n')
print("Classification report:")
print(report)


# 4.2  SVM

# In[80]:


# create a logistic regression model
SVMmodel = svm.SVC()
y_pred = cross_val_predict(SVMmodel, X, y, cv=5)

# Compute the performance metrics for the predicted target values and the true target values
cm = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)
auc = roc_auc_score(y, y_pred)

# Print out the performance metrics
print("Confusion matrix:")
print(cm)
print('\n')
print("AUC:", auc)
print('\n')
print("Classification report:")
print(report)


# 4.3  KNN

# In[88]:


# Create a KNN classifier and find best K
def knnresult():
    maxauc=0
    bestk=1
    for i in range(1,20):
        
        KNNmodel = KNeighborsClassifier(n_neighbors=i)
        y_pred = cross_val_predict(KNNmodel, X, y, cv=5)
        auc = roc_auc_score(y, y_pred)
        
        if auc>maxauc:
            maxauc=auc
            bestk=i
    
    
    KNNmodel = KNeighborsClassifier(n_neighbors=bestk)
    y_pred = cross_val_predict(KNNmodel, X, y, cv=5)

    # Compute the performance metrics for the predicted target values and the true target values
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    print('the best k is',bestk,'\n')
    # Print out the performance metrics
    print("Confusion matrix:")
    print(cm)
    print('\n')
    print("AUC:", auc)
    print('\n')
    print("Classification report:")
    print(report)
    
knnresult()


# For all 3 models, we are using the same 8 features: 'Pclass', 'SibSp', 'Parch', 'Gender', 'Age', 'S', 'Q', 'C'. 
# By comparing these three models we can see that logistic regression model performs the best. Thus we use it to predict on our testing set.

# # 5.Prediction for Training Set

# In[90]:


X_test = test[['Pclass', 'SibSp', 'Parch', 'Gender', 'Age', 'S', 'Q', 'C']]
#Fit the Model
X = train[['Pclass', 'SibSp', 'Parch', 'Gender', 'Age', 'S', 'Q', 'C']]
y = train['Survived']
Logisticmodel.fit(X, y)
# Make predictions on the test data
y_test_pred = Logisticmodel.predict(X_test)


# In[93]:


#Add predictions to df and export as xlsx
result=pd.read_csv('test.csv',header = 0, dtype={'Age': np.float64})
result.insert(1, 'Survived', y_test_pred)
result.to_excel('result.xlsx', index=False)

