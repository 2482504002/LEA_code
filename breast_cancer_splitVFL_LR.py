from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn
import numpy as np
from copy import deepcopy
import math
import torch
from torch.nn.functional import cosine_similarity


n_features = 28
w_c = np.zeros((n_features, 1))
w_s = np.zeros((30-n_features, 1))
w_t = np.zeros((2, 1))
data = load_breast_cancer()
X = data.data
scaler = StandardScaler()

# 拟合数据并进行归一化
X = scaler.fit_transform(X)
y = data.target
feature_indices_s = [18,19]
feature_indices_c = [i for i in range(30) if i not in feature_indices_s]
selected_features_c = X[:, feature_indices_c]
selected_features_s = X[:, feature_indices_s]
# 将数据集分为训练集和测试集
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(selected_features_c, y, test_size=0.2, random_state=42)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(selected_features_s, y, test_size=0.2, random_state=42)

x_c=np.concatenate([X_train_c,X_test_c],axis=0)
x_s=np.concatenate([X_train_s,X_test_s],axis=0)
yy=np.concatenate([y_train_s,y_test_s],axis=0).reshape(569,1)
learning_rate=0.01
num_iterations=1000
costs = []
m = X.shape[0]
for i in range(num_iterations):
   z_c = np.dot(X_train_c, w_c) 
   z_s = np.dot(X_train_s, w_s)
   z_cs=np.concatenate((z_c,z_s),axis=1)
   z=np.dot(z_cs, w_t) 
   y_h = 1.0/(1.0+pow(math.e,(-z)))
   y_h=y_h.reshape(455)
   dy=y_train_s-y_h
   Aa = np.tile(dy, (28,1)).T
   Ab = np.tile(dy, (2,1)).T
   At = np.tile(dy, (2,1)).T
   cost = -np.sum(y_train_s * np.log(y_h) + (1 - y_train_s) * np.log(1 - y_h)) / m
   
   # 计算梯度
   tmp_xa=np.multiply(Aa,X_train_c)
   tmp_suma=np.sum(tmp_xa,axis=0)
   tmp_xb=np.multiply(Ab, X_train_s)
   tmp_sumb=np.sum(tmp_xb,axis=0)
   tmp_xt=np.multiply(At, z_cs)
   tmp_sumt=np.sum(tmp_xt,axis=0)
   l_a=(tmp_suma/(-455)).reshape(28,1)
   l_b=(tmp_sumb/(-455)).reshape(2,1)
   l_t=(tmp_sumt/(-455)).reshape(2,1)
   
   w_s = w_s - learning_rate * l_b
   w_c = w_c - learning_rate * l_a
   w_t = w_t - learning_rate * l_t

   
   if i % 100 == 0:
      costs.append(cost)
      print("Cost after iteration %i: %f" % (i, cost))

z_c = np.dot(x_c, w_c)
z_s = np.dot(x_s, w_s)
z=np.concatenate((z_c,z_s),axis=1)
z=np.dot(z, w_t) 

a = 1.0/(1.0+pow(math.e,(-z)))
y_pred = (a > 0.5).astype(int)
print((y_pred==yy).sum()/len(yy))

