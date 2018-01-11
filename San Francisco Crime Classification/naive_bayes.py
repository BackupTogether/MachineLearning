# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 

# 载入数据
train = pd.read_csv('~/Documents/Files/oths/2018/ML/MachineLearning/San Francisco Crime Classification/train.csv', parse_dates = ['Dates'])
test = pd.read_csv('~/Documents/Files/oths/2018/ML/MachineLearning/San Francisco Crime Classification/test.csv', parse_dates = ['Dates'])

# print(train.head(10))
# print(test.head(10))

# 对 Category 进行编码，映射到整数
label = preprocessing.LabelEncoder()
crime = label.fit_transform(train.Category)

# 对训练集中星期几、地区、犯罪时间（小时、月、年，不采用日是因为星期几比日与犯罪情况更相关）三项特征进行二值化
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = pd.get_dummies(train.Dates.dt.hour)
month = pd.get_dummies(train.Dates.dt.month)
year = pd.get_dummies(train.Dates.dt.year)

# 矩阵横向合并
train_data = pd.concat([days, district, hour, month, year], axis=1)
# 训练集分类结果
train_data['crime'] = crime

# 测试集相同处理
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = pd.get_dummies(test.Dates.dt.hour)
month = pd.get_dummies(test.Dates.dt.month)
year = pd.get_dummies(test.Dates.dt.year)
test_data = pd.concat([days, district, hour, month, year], axis=1)

training, validation = train_test_split(train_data, test_size=0.4)

model = BernoulliNB()
feature_list = training.columns.tolist()
feature_list = feature_list[:len(feature_list) - 1]
print('Features: ', feature_list)
model.fit(training[feature_list], training['crime'])

predicted = np.array(model.predict_proba(validation[feature_list]))
print('BernoulliNB log-loss: %f' % (log_loss(validation['crime'], predicted)))

# 跑测试集并输出结果到 output-NB.csv
d1 = time.time()
model = BernoulliNB()
model.fit(train_data[feature_list], train_data['crime'])
test_predicted = np.array(model.predict_proba(test_data[feature_list]))
d2 = time.time()
print('BernoulliNB time: %f sec' %(d2-d1))
col_names = np.sort(train['Category'].unique())
print(col_names)
result = pd.DataFrame(data=test_predicted, columns=col_names)
result['Id'] = test['Id'].astype(int)
result.to_csv('~/Documents/Files/oths/2018/ML/MachineLearning/San Francisco Crime Classification/output-NB.csv', index=False)

model = LogisticRegression(C=0.1)
model.fit(training[feature_list], training['crime'])

predicted = np.array(model.predict_proba(validation[feature_list]))
print('LogisticRegression log-loss: %f' %(log_loss(validation['crime'], predicted)))

# 跑测试集并输出结果到 output-LR.csv
d1 = time.time()
model = LogisticRegression(C=0.1)
model.fit(train_data[feature_list], train_data['crime'])
test_predicted = np.array(model.predict_proba(test_data[feature_list]))
d2 = time.time()
print('LogisticRegression time: %f sec' %(d2-d1))
col_names = np.sort(train['Category'].unique())
print(col_names)
result = pd.DataFrame(data=test_predicted, columns=col_names)
result['Id'] = test['Id'].astype(int)
result.to_csv('~/Documents/Files/oths/2018/ML/MachineLearning/San Francisco Crime Classification/output-LR.csv', index=False)
