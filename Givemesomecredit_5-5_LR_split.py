import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import re as re
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.model_selection import train_test_split
import math
import torch
from torch.nn.functional import cosine_similarity

train_df = pd.read_csv('D:\datasets\Give_me_some_credit\cs-training.csv')
test_df = pd.read_csv('D:\datasets\Give_me_some_credit\cs-test.csv')

print (train_df.info())
print(train_df.head(5))

train_df.rename(columns={'Unnamed: 0':'ID'}, inplace=True)
test_df.rename(columns={'Unnamed: 0':'ID'}, inplace=True)

train_df.duplicated().value_counts()
test_df.duplicated().value_counts()
train_df.loc[train_df['age'] < 18]
train_df.loc[train_df['age'] == 0, 'age'] = train_df['age'].median()
working = train_df.loc[(train_df['age'] >= 18) & (train_df['age'] <= 60)]
senior = train_df.loc[(train_df['age'] > 60)]
working_income_mean = working['MonthlyIncome'].mean()
senior_income_mean = senior['MonthlyIncome'].mean()
print (working_income_mean)
print (senior_income_mean)
train_df['MonthlyIncome'] = train_df['MonthlyIncome'].replace(np.nan,train_df['MonthlyIncome'].mean())
train_df['NumberOfDependents'].fillna(train_df['NumberOfDependents'].median(), inplace=True)

X = train_df.drop(['SeriousDlqin2yrs', 'ID'],axis=1)
y = train_df['SeriousDlqin2yrs']
W = test_df.drop(['SeriousDlqin2yrs', 'ID'],axis=1)
z = test_df['SeriousDlqin2yrs']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=111)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

X_scale=X_scaled[:, 0:5]
Xt_scaled = scaler.fit_transform(X_test)
Xt_scale=Xt_scaled[:, 0:5]
X_scale1=X_scaled[:, 5:10]
# 使用K-Means算法进行聚类
# 由于是二分类问题，我们可以选择聚类数目为2
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scale)

# 获取聚类标签
labels01 = kmeans.labels_

kk=(labels01==y_train)
kkk=kk.sum()
print(f"聚类准确率：{kkk/len(y_train)}")
labels10=deepcopy(labels01)
for xx in range(len(labels10)):
   if labels10[xx]==0:
      labels10[xx]=1
   else:
      labels10[xx]=0

learning_rate=0.2
num_iterations=1000
X_train_c=X_scale
X_train_s=X_scale1

w_c = np.random.rand(5).reshape(5,1)
w_s = np.random.rand(5).reshape(5,1)

w_c11=deepcopy(w_c)
w_c22=deepcopy(w_c)

for i in range(num_iterations):
   z_c1 = np.dot(X_train_c, w_c11) 
   y_h = 1.0/(1.0+pow(math.e,(-z_c1)))
   y_h=y_h.reshape(112500)
   dy=labels01-y_h
   Aa = np.tile(dy, (5,1)).T
   cost = -np.sum(y_train * np.log(y_h) + (1 - y_train) * np.log(1 - y_h)) / 112500
   # 计算梯度
   tmp_xa=np.multiply(Aa,X_train_c)
   tmp_suma=np.sum(tmp_xa,axis=0)
   l_a=(tmp_suma/(-112500)).reshape(5,1)
   w_c11 = w_c11 - learning_rate * l_a
   if i % 100 == 0:
      
      print("Cost after iteration %i: %f" % (i, cost))
for i in range(num_iterations):
   z_c2 = np.dot(X_train_c, w_c22) 
   y_h = 1.0/(1.0+pow(math.e,(-z_c2)))
   y_h=y_h.reshape(112500)
   dy=labels10-y_h
   Aa = np.tile(dy, (5,1)).T
   epsilon = 1e-12
   clipped_y_h = np.clip(y_h, epsilon, 1 - epsilon)
   cost = -np.sum(y_train * np.log(clipped_y_h) + (1 - y_train) * np.log(1 - clipped_y_h)) / 112500
   # 计算梯度
   tmp_xa=np.multiply(Aa,X_train_c)
   tmp_suma=np.sum(tmp_xa,axis=0)
   l_a=(tmp_suma/(-112500)).reshape(5,1)
   w_c22= w_c22 - learning_rate * l_a
   if i % 100 == 0:

      print("Cost after iteration %i: %f" % (i, cost))



costs = []
m = X.shape[0]

for i in range(num_iterations):
   z_c = np.dot(X_train_c, w_c) 
   z_s = np.dot(X_train_s, w_s)
   z=np.concatenate([z_c+z_s],axis=1)
   y_h = 1.0/(1.0+pow(math.e,(-z)))
   y_h=y_h.reshape(112500)
   dy=y_train-y_h
   Aa = np.tile(dy, (5,1)).T
   Ab = np.tile(dy, (5,1)).T
   cost = -np.sum(y_train * np.log(y_h) + (1 - y_train) * np.log(1 - y_h)) / m
   
   # 计算梯度
   tmp_xa=np.multiply(Aa,X_train_c)
   tmp_suma=np.sum(tmp_xa,axis=0)


   tmp_xb=np.multiply(Ab, X_train_s)
   tmp_sumb=np.sum(tmp_xb,axis=0)
   l_a=(tmp_suma/(-112500)).reshape(5,1)

   l_b=(tmp_sumb/(-112500)).reshape(5,1)
   
   w_s = w_s - learning_rate * l_b
   w_c = w_c - learning_rate * l_a

   if i % 100 == 0:
      costs.append(cost)
      print("Cost after iteration %i: %f" % (i, cost))
      
w_c=torch.from_numpy(w_c)
w_c11=torch.from_numpy(w_c11)
w_c22=torch.from_numpy(w_c22)
w_c=w_c.view(w_c.numel())
w_c11=w_c11.view(w_c11.numel())
w_c22=w_c22.view(w_c22.numel())
c1=cosine_similarity(w_c, w_c11,dim=0)
c2=cosine_similarity(w_c, w_c22,dim=0)
print("c1: ",c1)
print("c1: ",c2)

if c1>c2:
   print("Get attack Model: model_1.")
   attack_model=w_c11
else:
   print("Get attack Model: model_2.")
   attack_model=w_c22
   
z_c = np.dot(Xt_scale, attack_model)
a = 1.0/(1.0+pow(math.e,(-z_c)))
y_pred = (a > 0.5).astype(int)
print((y_pred==y_test).sum()/len(y_test))

