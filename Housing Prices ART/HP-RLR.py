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
from matplotlib.ticker import FuncFormatter, MultipleLocator
import seaborn as sns
pd.options.display.max_columns = 50
sns.set_style('whitegrid')
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, LassoLarsCV, ElasticNet, Lasso
from sklearn.cross_validation import cross_val_score
import xgboost as xgb

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

#Define function for determining Tuning Parameter
def cv_error(model):
    cve= np.sqrt(-cross_val_score(model, X, Y, scoring="mean_squared_error", cv = 5))
    return(cve)
    
#Ridge
a_ridge = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cvScores_ridge = [cv_error(Ridge(alpha = alpha)).mean() for alpha in a_ridge]
cvScores_ridge = pd.Series(cvScores_ridge, index = a_ridge)
minimum_error = cvScores_ridge.min()

#Lasso
a_lasso = [1, 0.1, 0.001, 0.0005]
cvScores_lasso = [cv_error(Lasso(alpha = alpha)).mean() for alpha in a_lasso]
cvScores_lasso = pd.Series(cvScores_lasso, index = a_lasso)
minimum_error_lasso = cvScores_lasso.min()

#Plots
figure, (ax1, ax2) = plt.subplots(1,2,figsize = (17,5))
ax1.plot(cvScores_ridge)
ax1.set_title("Cross Validation (Ridge)")
ax1.set_xlabel("Regularization Parameter (alpha)")
ax1.set_ylabel("Error")
ax2.plot(cvScores_lasso)
ax2.set_title("Cross Validation (Lasso)")
ax2.set_xlabel("Regularization Parameter (alpha)")
ax2.set_ylabel("Error")

print("Minimum Error for Ridge Model: ", minimum_error)
print("Minimum Error for Lasso Model: ", minimum_error_lasso)
def ord_to_char(v, p=None):
    return chr(int(v))
    
#Picking up Ridge Model & figuring 10 most useful and 10 least useful parameters for Housing Price Prediction
ridgeReg = Ridge()
ridgeReg.fit(X,Y)
coef = pd.Series(ridgeReg.coef_, index = X.columns)
relevant_Coeff = coef.sort_values().tail(10)
irrelevant_Coeff = coef.sort_values().head(10)

#Plots
plt.figure(figsize=(20,10))
relevant_Coeff.plot(kind = "barh", title="Most Relevant Aspects of a House")

plt.figure(figsize=(20,10))
irrelevant_Coeff.plot(kind = 'barh', title="Least Relevant Aspects of a House")

#Remaining Feature Set
plt.figure(figsize= (50,10))
preds = pd.DataFrame({"Predicted":ridgeReg.predict(X), "true":Y})
preds["Difference"] = preds["true"] - preds["Predicted"]
preds.plot(x = "Predicted", y = "Difference",kind = "scatter", title = "Residual Features")

print (ridgeReg.score(X,Y))