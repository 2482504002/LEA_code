from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
import sklearn
import numpy as np
from copy import deepcopy
import math
import torch
from torch.nn.functional import cosine_similarity
import random
accc=0
lun=10
for _ in range(lun):
   n_features = 28
   w_c = np.random.randn(n_features, 1)
   w_s = np.random.randn(30-n_features, 1)
   w_t = np.abs(np.random.randn(2, 1))
   


   w_c1=deepcopy(w_c)
   w_c2=deepcopy(w_c)

   data = load_breast_cancer()
   X = data.data
   scaler = StandardScaler()
   X = scaler.fit_transform(X)
   y = data.target
   feature_indices_c = random.sample([i for i in range(30)], n_features)
   feature_indices_s = [i for i in range(30) if i not in feature_indices_c]
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
      y_h1 = 1.0/(1.0+pow(math.e,(-z_c1)))
      y_h1=y_h1.reshape(455)
      dy1=labels01-y_h1
      dyy1 = np.tile(dy1, (n_features,1)).T
      epsilon = 1e-12
      clipped_y_h = np.clip(y_h1, epsilon, 1 - epsilon)
      cost = -np.sum(labels01 * np.log(clipped_y_h) + (1 - labels01) * np.log(1 - clipped_y_h)) / 455
      # 计算梯度
      tmp_x1=np.multiply(dyy1,X_train_c)
      tmp_sum1=np.sum(tmp_x1,axis=0)
      l_1=(tmp_sum1/(-455)).reshape(n_features,1)
      if i == 0:
         deta1=deepcopy(torch.from_numpy(l_1))
      w_c1 = w_c1 - learning_rate * l_1
      if i % 100 == 0:
         print("Cost after iteration %i: %f" % (i, cost))
         
   p_c1 = np.dot(X_test_c, w_c1)
   a = 1.0/(1.0+pow(math.e,(-p_c1)))
   y_pred = (a > 0.5).astype(int).reshape(y_test_s.shape[0])
   panduan=y_pred==y_test_s
   acc=panduan.sum()/len(y_test_s)
   print("w1Attack acc: ",acc)

   for i in range(num_iterations):
      z_c2 = np.dot(X_train_c, w_c2) 
      y_h2 = 1.0/(1.0+pow(math.e,(-z_c2)))
      y_h2=y_h2.reshape(455)
      dy2=labels10-y_h2
      dyy2 = np.tile(dy2, (n_features,1)).T
      epsilon = 1e-12
      clipped_y_h = np.clip(y_h2, epsilon, 1 - epsilon)
      cost = -np.sum(labels10 * np.log(clipped_y_h) + (1 - labels10) * np.log(1 - clipped_y_h)) / 455
      # 计算梯度
      tmp_x2=np.multiply(dyy2,X_train_c)
      tmp_sum2=np.sum(tmp_x2,axis=0)
      l_2=(tmp_sum2/(-455)).reshape(n_features,1)
      if i == 0:
         deta2=deepcopy(torch.from_numpy(l_2))
      w_c2= w_c2 - learning_rate * l_2
      if i % 100 == 0:
         print("Cost after iteration %i: %f" % (i, cost))
   end=time.time()
   print(f"5class Time cost: {end-start} s")

   p_c2 = np.dot(X_test_c, w_c2)
   a = 1.0/(1.0+pow(math.e,(-p_c2)))
   y_pred = (a > 0.5).astype(int).reshape(y_test_s.shape[0])
   panduan=y_pred==y_test_s
   acc=panduan.sum()/len(y_test_s)
   print("w2Attack acc: ",acc)

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
      dyya = np.tile(dy*w_t[0], (n_features,1)).T
      dyyb = np.tile(dy*w_t[1], (30-n_features,1)).T
      dyyt = np.tile(dy, (2,1)).T
      cost = -np.sum(y_train_s * np.log(y_h) + (1 - y_train_s) * np.log(1 - y_h)) / m
      
      # 计算梯度
      tmp_xa=np.multiply(dyya,X_train_c)
      tmp_suma=np.sum(tmp_xa,axis=0)
      tmp_xb=np.multiply(dyyb, X_train_s)
      tmp_sumb=np.sum(tmp_xb,axis=0)
      
      tmp_xt=np.multiply(dyyt, z_cs)
      tmp_sumt=np.sum(tmp_xt,axis=0)
      l_a=(tmp_suma/(-455)).reshape(n_features,1)
      if i == 0:
         detaa=deepcopy(torch.from_numpy(l_a))
      l_b=(tmp_sumb/(-455)).reshape(30-n_features,1)
      l_t=(tmp_sumt/(-455)).reshape(2,1)

      w_s = w_s - learning_rate * l_b
      w_c = w_c - learning_rate * l_a
      w_t = w_t - learning_rate * l_t


      if i % 100 == 0:
         costs.append(cost)
         print("Cost after iteration %i: %f" % (i, cost))
   
   z_c = np.dot(X_test_c, w_c) 
   z_s = np.dot(X_test_s, w_s)
   z_cs=np.concatenate((z_c,z_s),axis=1)
   z=np.dot(z_cs, w_t) 
   a = 1.0/(1.0+pow(math.e,(-z)))
   y_pred = (a > 0.5).astype(int).reshape(y_test_s.shape[0])
   panduan=y_pred==y_test_s
   acc=panduan.sum()/len(y_test_s)
   print("acc: ",acc)

   w_c=torch.from_numpy(w_c)
   w_c1=torch.from_numpy(w_c1)
   w_c2=torch.from_numpy(w_c2)
   w_cc=w_c.view(w_c.numel())
   w_c11=w_c1.view(w_c1.numel())
   w_c22=w_c2.view(w_c2.numel())
   c1=cosine_similarity(w_cc, w_c11,dim=0)
   c2=cosine_similarity(w_cc, w_c22,dim=0)
   l1=abs(torch.sum(w_cc-w_c11))
   l2=abs(torch.sum(w_cc-w_c22))
   
   print("c1: ",c1,l1)
   print("c2: ",c2,l2)
   
   c1=cosine_similarity(detaa, deta1,dim=0)
   c2=cosine_similarity(detaa, deta2,dim=0)
      
   print("c1: ",c1)
   print("c2: ",c2)

   if c1>c2:
      print("Get attack Model: model_1.")
      attack_model=w_c1
   else:
      print("Get attack Model: model_2.")
      attack_model=w_c2


   z_c1 = np.dot(X_test_c, attack_model)
   
   a = 1.0/(1.0+pow(math.e,(-z_c1)))
   y_pred = (a > 0.5).astype(int).reshape(y_test_s.shape[0])
   panduan=y_pred==y_test_s
   acc=panduan.sum()/len(y_test_s)
   print("Attack acc: ",acc)

   if acc>0.5:
      accc+=1
print("Get right attack model: ",accc/lun)