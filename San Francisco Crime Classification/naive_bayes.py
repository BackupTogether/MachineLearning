#!/anaconda2/bin/python
# -*- coding: utf-8 -*-

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

# print train.head(10)
# print test.head(10)

# 对 Category 进行编码，映射到整数
label = preprocessing.LabelEncoder()
crime = label.fit_transform(train.Category)

# 对日期、星期几、地区三项特征进行二值化
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = pd.get_dummies(train.Dates.dt.hour)

# 抽取训练集中的
train_data = pd.concat([days, district, hour], axis=1)
train_data['crime'] = crime

days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = pd.get_dummies(test.Dates.dt.hour)
test_data = pd.concat([days, district, hour], axis=1)

training, validation = train_test_split(train_data, test_size=0.4)

model = BernoulliNB()
feature_list = training.columns.tolist()
feature_list = feature_list[:len(feature_list) - 1]
print '选取的特征列：', feature_list
model.fit(training[feature_list], training['crime'])

predicted = np.array(model.predict_proba(validation[feature_list]))
print "朴素贝叶斯log损失为 %f" % (log_loss(validation['crime'], predicted))

model = LogisticRegression(C=0.1)
model.fit(training[feature_list], training['crime'])

predicted = np.array(model.predict_proba(validation[feature_list]))
print "逻辑回归log损失为 %f" %(log_loss(validation['crime'], predicted))
