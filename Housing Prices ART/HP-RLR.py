# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 23:37:01 2016

@author: sominwadhwa
"""

#Imports
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_columns = 50
sns.set_style('whitegrid')


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder

#importing data
train_DF = pd.read_csv('train.csv')
test_DF = pd.read_csv('test.csv')
train_DF.head()

#Plot a general distribution of the prices (Fixing the skewed meytSkewed Metric)
figure, (ax1, ax2) = plt.subplots(1,2,figsize = (17,5))
sns.distplot(train_DF['SalePrice'], kde = False, ax = ax1, bins = 100)
sns.distplot(np.log1p(train_DF["SalePrice"]), kde = False, axlabel = 'Normalized Sales Price', ax = ax2, bins = 100)
train_DF['SalePrice'] = np.log1p(train_DF["SalePrice"])

#fixing other numeric skewed metrics
numeric_features_train = train_DF.dtypes[train_DF.dtypes != 'object'].index
numeric_features_test = test_DF.dtypes[train_DF.dtypes != 'object'].index

skewed_features_train = train_DF[numeric_features_train].apply(lambda x: skew(x))
skewed_features_test = test_DF[numeric_features_test].apply(lambda x: skew(x))

skewed_features_train = skewed_features_train[skewed_features_train > 0.75]
skewed_features_test = skewed_features_test[skewed_features_test > 0.75]
skewed_features_train = skewed_features_train.index
skewed_features_test = skewed_features_test.index

train_DF[skewed_features_train] = np.log1p(train_DF[skewed_features_train])
test_DF[skewed_features_test] = np.log1p(test_DF[skewed_features_test])

train_DF.head()

#Getting dummies for all the non numeric data
train_DF = pd.get_dummies(train_DF)
test_DF = pd.get_dummies(test_DF)

#Fill in empty values with mean of each column
train_DF = train_DF.fillna(train_DF.mean())
test_DF = test_DF.fillna(test_DF.mean())

#Classifying training and test data
X = train_DF.drop(['Id','SalePrice'], axis = 1, inplace = False)
Y = train_DF['SalePrice']
X_test = test_DF.drop('Id', axis = 1, inplace = False)