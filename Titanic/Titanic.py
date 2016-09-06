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
titanic_DF = pd.read_csv("/Users/sominwadhwa/Desktop/Kaggle/A journey through Titanic/train.csv", dtype = {"Age": np.float64},)
titanic_testDF = pd.read_csv("/Users/sominwadhwa/Desktop/Kaggle/A journey through Titanic/test.csv", dtype = {"Age": np.float64}, )

print (titanic_DF.head())  
print (titanic_DF.info())
print ("----------------------------")
print (titanic_testDF.info())