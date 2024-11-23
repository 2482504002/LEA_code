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


n_features = 19
w_c = np.random.rand(n_features).reshape(n_features,1)
w_s = np.random.rand(2).reshape(2,1)
w_c1 = np.random.rand(9).reshape(9,1)


w_c11=deepcopy(w_c)
w_c22=deepcopy(w_c)

data = load_breast_cancer()
X = data.data
scaler = StandardScaler()

# 拟合数据并进行归一化

X = scaler.fit_transform(X)
y = data.target
feature_indices_s = [18,19]
feature_indices_c1 = [1,3,5,7,9,11,13,15,17]
feature_indices_c = [i for i in range(30) if i not in feature_indices_s and i not in feature_indices_c1]
selected_features_c = X[:, feature_indices_c]
selected_features_c1 = X[:, feature_indices_c1]
selected_features_s = X[:, feature_indices_s]
# 将数据集分为训练集和测试集
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(selected_features_c, y, test_size=0.2, random_state=42)
X_train_c1, X_test_c1, y_train_c1, y_test_c1 = train_test_split(selected_features_c1, y, test_size=0.2, random_state=42)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(selected_features_s, y, test_size=0.2, random_state=42)


yy=np.concatenate([y_train_s,y_test_s],axis=0).reshape(569,1)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_c)

labels01 = kmeans.labels_
labels10=deepcopy(labels01)
#默认第一簇标签排序是0,1
kk=(labels01==y_train_c)
kkk=kk.sum()
acc01=kkk/len(y_train_c)
print("acc of labels01: ",acc01)

#第二簇标签排序是1,0
for xx in range(len(labels10)):
   if labels10[xx]==0:
      labels10[xx]=1
   else:
      labels10[xx]=0
kk=(labels10==y_train_c)
kkk=kk.sum()
acc10=kkk/len(y_train_c)
print("acc of labels10: ",acc10)


learning_rate=0.01
num_iterations=1000
for i in range(num_iterations):
   z_c1 = np.dot(X_train_c, w_c11) 
   y_h = 1.0/(1.0+pow(math.e,(-z_c1)))
   y_h=y_h.reshape(455)
   dy=labels01-y_h
   Aa = np.tile(dy, (n_features,1)).T
   cost = -np.sum(y_train_s * np.log(y_h) + (1 - y_train_s) * np.log(1 - y_h)) / 455
   # 计算梯度
   tmp_xa=np.multiply(Aa,X_train_c)
   tmp_suma=np.sum(tmp_xa,axis=0)
   l_a=(tmp_suma/(-455)).reshape(n_features,1)
   w_c11 = w_c11 - learning_rate * l_a
   if i % 100 == 0:
      
      print("Cost after iteration %i: %f" % (i, cost))
for i in range(num_iterations):
   z_c2 = np.dot(X_train_c, w_c22) 
   y_h = 1.0/(1.0+pow(math.e,(-z_c2)))
   y_h=y_h.reshape(455)
   dy=labels10-y_h
   Aa = np.tile(dy, (n_features,1)).T
   epsilon = 1e-12
   clipped_y_h = np.clip(y_h, epsilon, 1 - epsilon)
   cost = -np.sum(y_train_s * np.log(clipped_y_h) + (1 - y_train_s) * np.log(1 - clipped_y_h)) / 455
   # 计算梯度
   tmp_xa=np.multiply(Aa,X_train_c)
   tmp_suma=np.sum(tmp_xa,axis=0)
   l_a=(tmp_suma/(-455)).reshape(n_features,1)
   w_c22= w_c22 - learning_rate * l_a
   if i % 100 == 0:
      
      print("Cost after iteration %i: %f" % (i, cost))


costs = []
m = X.shape[0]
for i in range(num_iterations):
   z_c = np.dot(X_train_c, w_c) 
   z_c1 = np.dot(X_train_c1, w_c1) 
   z_s = np.dot(X_train_s, w_s)
   z=np.concatenate([z_c+z_s+z_c1],axis=1)
   y_h = 1.0/(1.0+pow(math.e,(-z)))
   y_h=y_h.reshape(455)
   dy=y_train_s-y_h
   Aa = np.tile(dy, (n_features,1)).T
   Aa1 = np.tile(dy, (9,1)).T
   Ab = np.tile(dy, (2,1)).T
   cost = -np.sum(y_train_s * np.log(y_h) + (1 - y_train_s) * np.log(1 - y_h)) / m
   
   # 计算梯度
   tmp_xa=np.multiply(Aa,X_train_c)
   tmp_suma=np.sum(tmp_xa,axis=0)
   tmp_xa1=np.multiply(Aa1,X_train_c1)
   tmp_suma1=np.sum(tmp_xa1,axis=0)
   tmp_xb=np.multiply(Ab, X_train_s)
   tmp_sumb=np.sum(tmp_xb,axis=0)
   l_a=(tmp_suma/(-455)).reshape(n_features,1)
   l_a1=(tmp_suma1/(-455)).reshape(9,1)
   l_b=(tmp_sumb/(-455)).reshape(2,1)
   
   w_s = w_s - learning_rate * l_b
   w_c = w_c - learning_rate * l_a
   w_c1 = w_c1 - learning_rate * l_a1

   
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
if acc10<acc01:
   print("Target attack Model: model_1.")
else:
   print("Target attack Model: model_2.")

if c1>c2:
   print("Get attack Model: model_1.")
   attack_model=w_c11
else:
   print("Get attack Model: model_2.")
   attack_model=w_c22
   
z_c = np.dot(X_test_c, attack_model)
a = 1.0/(1.0+pow(math.e,(-z_c)))
y_pred = (a > 0.5).astype(int)
print((y_pred==y_test_s).sum()/len(y_test_s))

