# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.model_selection import KFold
import lightgbm as lgbm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
macro_df = pd.read_csv('macro.csv')
id_test = test_df.id

print(train_df.shape, test_df.shape)

def runLGBM(train_X, train_y, test_X, seed_val=42):
    params = {
        'boosting_type': 'gbdt', 'objective': 'regression', 'nthread': -1, 'verbose': 1,        'num_leaves': 31, 'learning_rate': 0.05, 'max_depth': -1,
        'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.6, 
        'reg_alpha': 1, 'reg_lambda': 0.001, 'metric': 'rmse',
        'min_split_gain': 0.5, 'min_child_weight': 1, 'min_child_samples': 10, 'scale_pos_weight': 1}
    
    #kf = KFold(n_splits=5, shuffle=True, random_state=seed_val)
    pred_test_y = np.zeros(test_X.shape[0])
    
    train_set = lgbm.Dataset(train_X, train_y, silent=True)
        
    model = lgbm.train(params, train_set=train_set, num_boost_round=300)
    pred_test_y = model.predict(test_X, num_iteration = model.best_iteration)
        
    return pred_test_y , model

#features preparation
#Time
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
train_df["year"] = train_df["timestamp"].dt.year
test_df["year"] = test_df["timestamp"].dt.year
train_df["month"] = train_df["timestamp"].dt.month
test_df["month"] = test_df["timestamp"].dt.month
train_df["day"] = train_df["timestamp"].dt.day
test_df["day"] = test_df["timestamp"].dt.day
train_df["hour"] = train_df["timestamp"].dt.hour
test_df["hour"] = test_df["timestamp"].dt.hour

joined =pd.concat([train_df,test_df])

cat_cols = [x for x in joined.columns if joined[x].dtype == 'object']

for col in cat_cols:
    joined.loc[:,col] = pd.factorize(joined[col], sort=True)[0]

train_df = joined[joined['price_doc'].notnull()]
test_df = joined[joined['price_doc'].isnull()]

del joined

print(train_df.shape, test_df.shape)

train_y = np.log(train_df["price_doc"])
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
test_X = test_df.drop(["id", "timestamp"], axis=1)

predictions, model = runLGBM(train_X, train_y, test_X, seed_val=42)

pd.DataFrame({'id': id_test, 'price_doc': np.exp(predictions)}).to_csv('submission.csv', index=False)
