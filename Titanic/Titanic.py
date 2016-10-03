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
train_DF = pd.read_csv("train.csv", dtype = {"Age": np.float64}, )
test_DF = pd.read_csv("test.csv", dtype = {"Age": np.float64}, )

print (train_DF.head())  
print (test_DF.info())
print ("----------------------------")

#Dropping Unnecessary Data
train_DF = train_DF.drop(['PassengerId','Name','Ticket'], axis = 1, inplace = False)
test_DF = test_DF.drop(['Name','Ticket'], axis = 1, inplace = False)


#Checking probability of survival based on the place from where a passanger embarks
sns.factorplot(x = 'Embarked', y = 'Survived', data=train_DF, size = 3, aspect = 3)
figure, (ax1,ax2, ax3) = plt.subplots(1,3,figsize=(10,5))
sns.countplot(x = 'Embarked', data = train_DF, ax = ax1)
sns.countplot(x = 'Survived', hue = 'Embarked', data = train_DF, ax = ax2)
embark_perc = train_DF[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,ax=ax3)
embark_perc.head()
sns.countplot(x = 'Embarked', data = train_DF)
sns.countplot(x = 'Survived', hue = 'Embarked', data = train_DF)
#Introducing dummies for Embarked
embark_dummy = pd.get_dummies(train_DF['Embarked'])
train_DF = train_DF.join(embark_dummy) #May or may not choose to drop 'S' here (Due to lower chances of survival)
embark_dummy_test = pd.get_dummies(test_DF['Embarked'])
test_DF = test_DF.join(embark_dummy_test)
train_DF.drop(['Embarked'], axis = 1, inplace = True)
test_DF.drop(['Embarked'], axis = 1, inplace = True)
train_DF['C'] = train_DF['C'].astype(int)
train_DF['Q'] = train_DF['Q'].astype(int)
train_DF['S'] = train_DF['S'].astype(int)
train_DF['C'] = train_DF['C'].astype(int)
train_DF['Q'] = train_DF['Q'].astype(int)
train_DF['S'] = train_DF['S'].astype(int)
print (train_DF.head())

#Feature: Fare 
""" Since fare is one such obvious feature it is 
safe to say to include in the feature without much
thought """
#Perform Cleanup
test_DF['Fare'].fillna(test_DF['Fare'].median(), inplace = True)
train_DF['Fare'] = train_DF['Fare'].astype(int)
test_DF['Fare'] = test_DF['Fare'].astype(int)
#Fetch Fare Information
survived_fare = train_DF['Fare'][train_DF['Survived'] == 1]
not_survived_fare = train_DF['Fare'][train_DF['Survived'] == 0]
#Extract Metrics
get_avg_fare = DataFrame([survived_fare.mean(), not_survived_fare.mean()])
get_std_fare = DataFrame([survived_fare.std(), not_survived_fare.std()])
#Plot
train_DF['Fare'].plot(kind = 'hist', figsize = (15,3), bins = 100, xlim = (0,50))

facet = sns.FacetGrid(train_DF, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_DF['Age'].max()))
facet.add_legend()
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = train_DF[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)

#Feature: Cabin
train_DF.drop("Cabin",axis=1,inplace=True)
test_DF.drop("Cabin",axis=1,inplace=True)

#Family i.e Parch + Siblings
train_DF['Family'] =  train_DF["Parch"] + train_DF["SibSp"]
train_DF['Family'].loc[train_DF['Family'] > 0] = 1
train_DF['Family'].loc[train_DF['Family'] == 0] = 0
test_DF['Family'] =  test_DF["Parch"] + test_DF["SibSp"]
test_DF['Family'].loc[test_DF['Family'] > 0] = 1
test_DF['Family'].loc[test_DF['Family'] == 0] = 0


train_DF.drop(["Parch", "SibSp"], axis = 1, inplace = True)
test_DF.drop(["Parch", "SibSp"], axis = 1, inplace = True)

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,5))
sns.countplot(x = 'Family', data = train_DF, ax = ax1)
ax1.set_xticklabels(['w/ Family','w/o Family'], rotation =0)

family_survival = train_DF[['Family', 'Survived']].groupby(['Family'], as_index = 0).mean()
sns.barplot(x = 'Family', y = 'Survived', data = family_survival, ax = ax2)
ax2.set_xticklabels(['w/ Family','w/o Family'], rotation =0)