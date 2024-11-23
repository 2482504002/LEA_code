from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn
import numpy as np
from copy import deepcopy
import torch
from torch.nn.functional import cosine_similarity
def compare_layers_cosine_similarity1(layer1, layer2):
   weight1 = layer1.weight.data
   weight2 = layer2.weight.data
   if layer1.bias is not None and layer2.bias is not None:
      weight1=torch.cat(((layer1.bias.data).unsqueeze(1),weight1 ), dim=1)
      weight2=torch.cat(((layer2.bias.data).unsqueeze(1),weight2 ), dim=1)
   cos_s=0
   for i in range(weight1.shape[0]):
      cos=cosine_similarity(weight1[i], weight2[i],dim=0)
      cos_s+=cos
   cos_s=cos_s/weight1.shape[0]

   return cos_s
def compare_layers_cosine_similarity(layer1, layer2):
   weight1 = layer1.weight.data
   weight2 = layer2.weight.data
   weight1_flattened = weight1.view(weight1.numel())
   weight2_flattened = weight2.view(weight2.numel())
   l1=abs(torch.sum(weight1_flattened-weight2_flattened))
   l2=torch.norm(weight2_flattened)
   weight_similarity = cosine_similarity(weight1_flattened, weight2_flattened,dim=0)

   if layer1.bias is not None and layer2.bias is not None:
      
      b_layer1=torch.cat(((layer1.bias.data).unsqueeze(1),layer1.weight.data ), dim=1)
      b_layer2=torch.cat(((layer2.bias.data).unsqueeze(1),layer2.weight.data ), dim=1)
      cos_sim = cosine_similarity(b_layer1.view(-1), b_layer2.view(-1), dim=0)
      l1=abs(torch.sum(weight1_flattened-weight2_flattened))
      l2=torch.norm(b_layer2,2)
      # 将权重展平为一维向量

      bias_similarity = cosine_similarity(layer1.bias.data, layer2.bias.data,dim=0 )
   else:
      return weight_similarity,weight_similarity,weight_similarity,weight_similarity,l1,l2
   return bias_similarity,weight_similarity,cos_sim,(weight_similarity + bias_similarity) / 2,l1,l2
 
def load_data():
   # 加载乳腺癌数据集
   data = load_breast_cancer()
   X = data.data
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

   scaler=sklearn.preprocessing.StandardScaler()
   X_train_c=scaler.fit_transform(X_train_c)
   X_train_c1=scaler.fit_transform(X_train_c1)
   X_train_s=scaler.fit_transform(X_train_s)
   X_test_c=scaler.fit_transform(X_test_c)
   X_test_c1=scaler.fit_transform(X_test_c1)
   X_test_s=scaler.fit_transform(X_test_s)

   X_train_c=torch.from_numpy(X_train_c.astype(np.float32))
   X_train_c1=torch.from_numpy(X_train_c1.astype(np.float32))
   X_test_c=torch.from_numpy(X_test_c.astype(np.float32))
   X_test_c1=torch.from_numpy(X_test_c1.astype(np.float32))
   #y_train_c=torch.from_numpy(y_train_c.astype(np.float32))
   #y_test_c=torch.from_numpy(y_test_c.astype(np.float32))
   X_train_s=torch.from_numpy(X_train_s.astype(np.float32))
   X_test_s=torch.from_numpy(X_test_s.astype(np.float32))
   #y_train_s=torch.from_numpy(y_train_s.astype(np.float32))
   #y_test_s=torch.from_numpy(y_test_s.astype(np.float32))

   #y_train_c=y_train_c.view(y_train_c.shape[0],1)
   #y_test_c=y_test_c.view(y_test_c.shape[0],1)
   #y_train_s=y_train_s.view(y_train_s.shape[0],1)
   #y_test_s=y_test_s.view(y_test_s.shape[0],1)
   return X_train_c, X_test_c, y_train_c, y_test_c, X_train_s, X_test_s, y_train_s, y_test_s,X_train_c1, X_test_c1

class Logistic_Reg_model(torch.nn.Module):
   def __init__(self,no_input_features):
      super(Logistic_Reg_model,self).__init__()
      self.layer1=torch.nn.Linear(no_input_features,20)
      self.layer2=torch.nn.Linear(20,1)
   def forward(self,x):
      y_predicted=self.layer1(x)
      #y_predicted=torch.sigmoid(self.layer2(y_predicted))
      y_predicted=self.layer2(y_predicted)
      return y_predicted
 
class bottle_net(torch.nn.Module):
   def __init__(self,no_input_features):
      super(bottle_net,self).__init__()
      self.layer1=torch.nn.Linear(no_input_features,20,bias=False)
      
   def forward(self,x):
      y_predicted=self.layer1(x)
      return y_predicted
  
class simulate_net(torch.nn.Module):
   def __init__(self,no_input_features):
      super(simulate_net,self).__init__()
      self.layer1=torch.nn.Linear(no_input_features,20,bias=False)
      
      self.layer2=torch.nn.Linear(20,1,bias=False)
      self.layer2.weight.data=torch.clamp(self.layer2.weight.data, 0,1)
   def forward(self,x):
      y_predicted=self.layer1(x)
      #y_predicted=torch.sigmoid(self.layer2(y_predicted))
      y_predicted=torch.sigmoid(self.layer2(y_predicted))
      return y_predicted
 
class top_net(torch.nn.Module):
   def __init__(self,client_num):
      super(top_net,self).__init__()
      #self.layer1=torch.nn.Linear(no_input_features,20)
      self.layer2=torch.nn.Linear(client_num*20,1,bias=False)
      self.layer2.weight.data=torch.clamp(self.layer2.weight.data, 0,1)
   def forward(self,x):
      #y_predicted=torch.sigmoid(self.layer2(y_predicted))
      y_predicted=torch.sigmoid(self.layer2(x))
      return y_predicted

###################################上文无关###################################
criterion=torch.nn.BCELoss()
X_train_c, X_test_c, y_train_c, y_test_c, X_train_s, X_test_s, y_train_s, y_test_s,X_train_c1, X_test_c1=load_data()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_c)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

labels01 = kmeans.labels_
labels10=deepcopy(labels01)
#默认第一簇标签排序是0,1
kk=(labels01==y_train_c)
kkk=kk.sum()
print("acc of labels01: ",kkk/len(y_train_c))
labels01=torch.from_numpy(labels01.astype(np.float32))
labels01=labels01.view(labels01.shape[0],1)
#第二簇标签排序是1,0
for xx in range(len(labels10)):
   if labels10[xx]==0:
      labels10[xx]=1
   else:
      labels10[xx]=0
kk=(labels10==y_train_c)
kkk=kk.sum()
print("acc of labels10: ",kkk/len(y_train_c))
labels10=torch.from_numpy(labels10.astype(np.float32))
labels10=labels10.view(labels10.shape[0],1)

c_features=19
model_c=bottle_net(c_features)
model_c0=bottle_net(9)
model_s=bottle_net(2)
model_t=top_net(3)
model_c1=simulate_net(c_features)

model_c1.layer1=deepcopy(model_c.layer1)
#model_c1.layer2=deepcopy(model_t.layer2)
model_c2=deepcopy(model_c1)

optimizer_c1=torch.optim.SGD(model_c1.parameters(),lr=0.1)
optimizer_c2=torch.optim.SGD(model_c2.parameters(),lr=0.1)
number_of_epochs=1000
for epoch in range(number_of_epochs):
    y_prediction=model_c1(X_train_c)
    loss=criterion(y_prediction,labels01)
    optimizer_c1.zero_grad()
    loss.backward()
    optimizer_c1.step()
    if (epoch+1)%200 == 0:
        print('epoch:', epoch+1,',loss=',loss.item())
for epoch in range(number_of_epochs):
    y_prediction=model_c2(X_train_c)#from kmeans
    loss=criterion(y_prediction,labels10)
    optimizer_c2.zero_grad()
    loss.backward()
    optimizer_c2.step()
    
    if (epoch+1)%200 == 0:
        print('epoch:', epoch+1,',loss=',loss.item())

    

###################################下文无关###################################

#X_train_c has 28 features, X_train_s has 2 features
#X_train_c, X_test_c, y_train_c, y_test_c, X_train_s, X_test_s, y_train_s, y_test_s=load_data()
# 创建逻辑回归模型c
criterion=torch.nn.BCELoss()

#yy=torch.cat((y_train_s,y_test_s),dim=0)
with torch.no_grad():
    y_prediction1=model_c1(X_test_c)
    y_prediction2=model_c2(X_test_c)
    
    y_pred_class1=y_prediction1.round().reshape(114)
    y_pred_class1=np.array(y_pred_class1)
 
    y_pred_class2=y_prediction2.round().reshape(114)
    y_pred_class2=np.array(y_pred_class2)

    accuracy1=(y_test_s==y_pred_class1).sum()/len(y_test_s)
    accuracy2=(y_test_s==y_pred_class2).sum()/len(y_test_s)
    
    print("acc: ", accuracy1.item())
    print("acc: ", accuracy2.item())
    

#model_c=bottle_net(c_features)
optimizer_c=torch.optim.SGD(model_c.parameters(),lr=0.1)
optimizer_c0=torch.optim.SGD(model_c0.parameters(),lr=0.1)
#model_s=bottle_net(s_features)
optimizer_s=torch.optim.SGD(model_s.parameters(),lr=0.1)
optimizer_t=torch.optim.SGD(model_t.parameters(),lr=0.1)
number_of_epochs=1000

y_train_s=torch.from_numpy(y_train_s.astype(np.float32))
y_train_s=y_train_s.view(y_train_s.shape[0],1)
for epoch in range(number_of_epochs):
    y_prediction1=model_c(X_train_c)
    y_prediction2=model_s(X_train_s)
    y_prediction3=model_c0(X_train_c1)
    V=torch.cat((y_prediction1,y_prediction2,y_prediction3), dim=1)
    
    y_prediction =model_t(V)
    loss=criterion(y_prediction,y_train_s)#
    optimizer_c.zero_grad()
    optimizer_c0.zero_grad()
    optimizer_s.zero_grad()
    optimizer_t.zero_grad()
    loss.backward()
    optimizer_c.step()
    optimizer_c0.step()
    optimizer_s.step()
    optimizer_t.step()
    if (epoch+1)%200 == 0:
        print('epoch:', epoch+1,',loss=',loss.item())
with torch.no_grad():
    y_prediction1=model_c(X_test_c)
    y_prediction2=model_s(X_test_s)
    y_prediction3=model_c0(X_test_c1)
    V=torch.cat((y_prediction1,y_prediction2,y_prediction3), dim=1)
    y_pred =model_t(V)
    
    y_pred_class=y_pred.round().reshape(114)
    y_pred_class=np.array(y_pred_class)

    accuracy=(y_pred_class==y_test_s).sum()/len(y_pred_class)
    
    #accuracy=(y_pred_class.eq(y_test_s).sum())/float(y_test_s.shape[0])#y_test_c=y_test_s
    print("acc: ", accuracy.item())
   
bottle_layer1 = model_c1.layer1
bottle_layer2 = model_c2.layer1
bottle_layer = model_c.layer1
b1,w1,cc1,c1,l11,l12=compare_layers_cosine_similarity(bottle_layer,bottle_layer1)
b2,w2,cc2,c2,l21,l22=compare_layers_cosine_similarity(bottle_layer,bottle_layer2)
b3,w3,cc3,c3,l31,l32=compare_layers_cosine_similarity(bottle_layer1,bottle_layer2)
print(f"c1: {b1,w1,cc1,c1,l11,l12}")
print(f"c2: {b2,w2,cc2,c2,l21,l22}")
print(f"c3: {b3,w3,cc3,c3,l31,l32}")
'''
c3=compare_layers_cosine_similarity1(bottle_layer1,bottle_layer2)
print(f"c3: {c3}")
c1=compare_layers_cosine_similarity1(bottle_layer,bottle_layer1)
c2=compare_layers_cosine_similarity1(bottle_layer,bottle_layer2)

print(f"c1: {c1}")
print(f"c2: {c2}")

bottle_layer1 = model_c1.layer1
bottle_layer2 = model_c2.layer1
bottle_layer = model_c.layer1
b1,w1,cc1,c1=compare_layers_cosine_similarity(bottle_layer,bottle_layer1)
b2,w2,cc2,c2=compare_layers_cosine_similarity(bottle_layer,bottle_layer2)
b2,w2,cc3,c2=compare_layers_cosine_similarity(bottle_layer1,bottle_layer2)
print(f"c1: {cc1}")
print(f"c2: {cc2}")
print(f"c3: {cc3}")
'''
'''
bottle_layer1 = model_c1.layer00
bottle_layer2 = model_c2.layer00
bottle_layer = model_c.layer00
b1,w1,cc1,c1=compare_layers_cosine_similarity(bottle_layer,bottle_layer1)
b2,w2,cc2,c2=compare_layers_cosine_similarity(bottle_layer,bottle_layer2)
b2,w2,cc3,c2=compare_layers_cosine_similarity(bottle_layer1,bottle_layer2)
print(f"c1: {cc1}")
print(f"c2: {cc2}")
print(f"c3: {cc3}")'''



if l21>l11:
   print("Get attack Model: model_1.")
   attack_model=model_c1
else:
   print("Get attack Model: model_2.")   
   attack_model=model_c2

with torch.no_grad():
    y_pred=attack_model(X_test_c)
    y_pred_class=y_pred.round().reshape(114)
    y_pred_class=np.array(y_pred_class)
    accuracy=(y_pred_class==y_test_s).sum()/len(y_pred_class)
    print("Attack acc: ", accuracy.item())
