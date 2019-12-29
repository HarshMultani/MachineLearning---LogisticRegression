# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 11:47:53 2019

@author: 138709
"""

# Logistic Regression


# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values


# Plotting to see the relation between the variables
plt.scatter(dataset.iloc[:,2].values, dataset.iloc[:,4].values)
plt.title('Social Network')
plt.xlabel('Age')
plt.ylabel('Purchased')
plt.show()

plt.scatter(dataset.iloc[:,3].values, dataset.iloc[:,4].values)
plt.title('Social Network')
plt.xlabel('Estimates Salary')
plt.ylabel('Purchased')
plt.show()


# Split the dataset into train and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# Feature Scaling the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Fitting Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(X_train, Y_train)


# Predicting the test set results
Y_pred = model.predict(X_test)


# Making the confusion matrix
from sklearn.metrics import confusion_matrix
results = confusion_matrix(Y_test, Y_pred)
print(results)
