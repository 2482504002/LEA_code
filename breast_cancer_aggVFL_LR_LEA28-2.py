from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from copy import deepcopy
import math
import torch
from torch.nn.functional import cosine_similarity
from scipy.stats import laplace

def add_laplace_noise(data, epsilon, sensitivity=1):
    b =  epsilon/sensitivity
    noise = laplace.rvs(loc=0, scale=b, size=data.shape)
    noisy_data = data + noise
    return noisy_data

dp=0
epsilon=100
n_features = 28
w_c = np.random.rand(n_features).reshape(n_features,1)
w_s = np.random.rand(30-n_features).reshape(30-n_features,1)

w_c1=deepcopy(w_c)
w_c2=deepcopy(w_c)

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

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_c)

labels01 = kmeans.labels_
labels10=deepcopy(labels01)
#默认第一簇标签排序是0,1
kk=(labels01==y_train_c)
kkk=kk.sum()
print("acc of labels01: ",kkk/len(y_train_c))

#第二簇标签排序是1,0
for xx in range(len(labels10)):
   if labels10[xx]==0:
      labels10[xx]=1
   else:
      labels10[xx]=0
kk=(labels10==y_train_c)
kkk=kk.sum()
print("acc of labels10: ",kkk/len(y_train_c))

learning_rate=0.01
num_iterations=1000
import time
start=time.time()
for i in range(num_iterations):
   z_c1 = np.dot(X_train_c, w_c1) 
   y_h = 1.0/(1.0+pow(math.e,(-z_c1)))
   y_h=y_h.reshape(455)
   dy=labels01-y_h
   Aa = np.tile(dy, (28,1)).T
   cost = -np.sum(y_train_s * np.log(y_h) + (1 - y_train_s) * np.log(1 - y_h)) / 455
   # 计算梯度
   tmp_xa=np.multiply(Aa,X_train_c)
   tmp_suma=np.sum(tmp_xa,axis=0)
   l_a=(tmp_suma/(-455)).reshape(28,1)

   w_c1 = w_c1 - learning_rate * l_a
   if i==1:
      grad_1=deepcopy(l_a)
   if i % 100 == 0:
      print("Cost after iteration %i: %f" % (i, cost))
      
for i in range(num_iterations):
   z_c2 = np.dot(X_train_c, w_c2) 
   y_h = 1.0/(1.0+pow(math.e,(-z_c2)))
   y_h=y_h.reshape(455)
   dy=labels10-y_h
   Aa = np.tile(dy, (28,1)).T
   cost = -np.sum(y_train_s * np.log(y_h) + (1 - y_train_s) * np.log(1 - y_h)) / 455
   # 计算梯度
   tmp_xa=np.multiply(Aa,X_train_c)
   tmp_suma=np.sum(tmp_xa,axis=0)
   l_a=(tmp_suma/(-455)).reshape(28,1)
   w_c2= w_c2 - learning_rate * l_a
   if i==1:
      grad_2=deepcopy(l_a)
   if i % 100 == 0:
      print("Cost after iteration %i: %f" % (i, cost))
end=time.time()
print(f"5class Time cost: {end-start} s")
costs = []
m = X.shape[0]
for i in range(num_iterations):
   z_c = np.dot(X_train_c, w_c) 
   z_s = np.dot(X_train_s, w_s)
   z=z_c+z_s
   y_h = 1.0/(1.0+pow(math.e,(-z)))
   y_h=y_h.reshape(455)
   dy=y_train_s-y_h
   Aa = np.tile(dy, (28,1)).T
   Ab = np.tile(dy, (2,1)).T
   cost = -np.sum(y_train_s * np.log(y_h) + (1 - y_train_s) * np.log(1 - y_h)) / m
   
   # 计算梯度
   tmp_xa=np.multiply(Aa,X_train_c)
   tmp_suma=np.sum(tmp_xa,axis=0)
   tmp_xb=np.multiply(Ab, X_train_s)
   tmp_suma=np.sum(tmp_xa,axis=0)
   tmp_sumb=np.sum(tmp_xb,axis=0)
   l_a=(tmp_suma/(-455)).reshape(28,1)
   l_b=(tmp_sumb/(-455)).reshape(2,1)
   if i==1:
      grad_c=deepcopy(l_a)
   if dp==1:
      l_a=add_laplace_noise(l_a,epsilon)
   w_s = w_s - learning_rate * l_b
   w_c = w_c - learning_rate * l_a

   
   if i % 100 == 0:
      costs.append(cost)
      print("Cost after iteration %i: %f" % (i, cost))

z_c = np.dot(X_test_c, w_c)
z_s = np.dot(X_test_s, w_s)
z=z_c+z_s

a = 1.0/(1.0+pow(math.e,(-z)))
y_pred = (a > 0.5).astype(int).reshape(114)
print((y_pred==y_test_c).sum()/len(y_test_c))



w_c=torch.from_numpy(w_c)
w_c1=torch.from_numpy(w_c1)
w_c2=torch.from_numpy(w_c2)
w_c=w_c.view(w_c.numel())
w_c1=w_c1.view(w_c1.numel())
w_c2=w_c2.view(w_c2.numel())
c1=cosine_similarity(w_c, w_c1,dim=0)
c2=cosine_similarity(w_c, w_c2,dim=0)
print("cos1: ",c1)
print("cos2: ",c2)
if dp==1:
   grad_c=add_laplace_noise(grad_c,0.01*epsilon)
c1=cosine_similarity(torch.from_numpy(grad_c), torch.from_numpy(grad_1),dim=0)
c2=cosine_similarity(torch.from_numpy(grad_c), torch.from_numpy(grad_2),dim=0)
print("cos1: ",c1)
print("cos2: ",c2)
if c1>c2:
   print("Get attack Model: model_1.")
   attack_model=w_c1
else:
   print("Get attack Model: model_2.")
   attack_model=w_c2
z_c = np.dot(X_test_c, attack_model)
a = 1.0/(1.0+pow(math.e,(-z_c)))
y_pred = (a > 0.5).astype(int)
print("Attack model acc: ",(y_pred==y_test_s).sum()/len(y_test_s))

