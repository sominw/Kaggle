# -*- coding: utf-8 -*-
"""
A Journey through Titanic

Created on Mon Sep  5 23:39:36 2016

@author: sominwadhwa
"""
#imports
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

#Machine Learning Library Imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

#Fetch Data
train_DF = pd.read_csv("/Users/sominwadhwa/Desktop/Kaggle/Titanic/train.csv", dtype = {"Age": np.float64}, )
test_DF = pd.read_csv("/Users/sominwadhwa/Desktop/Kaggle/Titanic/test.csv", dtype = {"Age": np.float64}, )

print (train_DF.head())  
print (test_DF.info())
print ("----------------------------")

#Dropping Unnecessary Data
train_DF = train_DF.drop(['PassengerId','Name','Ticket'], axis = 1, inplace = False)
test_DF = test_DF.drop(['Name','Ticket'], axis = 1, inplace = False)

sns.factorplot(x='Embarked',y='Survived', data=train_DF)