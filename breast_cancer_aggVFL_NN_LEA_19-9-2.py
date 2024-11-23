from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn
import numpy as np
from copy import deepcopy
import torch
from torch.nn.functional import cosine_similarity

def compare_layers_cosine_similarity(layer1, layer2):
   weight1 = layer1.weight.data
   weight2 = layer2.weight.data
   b_layer1=torch.cat(((layer1.bias.data).unsqueeze(1),layer1.weight.data ), dim=1)
   b_layer2=torch.cat(((layer2.bias.data).unsqueeze(1),layer2.weight.data ), dim=1)
   cos_sim = cosine_similarity(b_layer1.view(-1), b_layer2.view(-1), dim=0)
   # 将权重展平为一维向量
   weight1_flattened = weight1.view(weight1.numel())
   weight2_flattened = weight2.view(weight2.numel())
    
   weight_similarity = cosine_similarity(weight1_flattened, weight2_flattened,dim=0)

   bias_similarity = cosine_similarity(layer1.bias.data, layer2.bias.data,dim=0 ) if layer1.bias is not None and layer2.bias is not None else 1
   return bias_similarity,weight_similarity,cos_sim,(weight_similarity + bias_similarity) / 2
 
def load_data():
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_indices_s = [18,19]
    feature_indices_c1 = [1,3,5,7,9,11,13,15,17]
    feature_indices_c = [i for i in range(30) if i not in feature_indices_s and i not in feature_indices_c1]
    
    selected_features_c = X[:, feature_indices_c]
    selected_features_s = X[:, feature_indices_s]
    selected_features_c1 = X[:, feature_indices_c1]
    # 将数据集分为训练集和测试集
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(selected_features_c, y, test_size=0.2, random_state=42)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(selected_features_s, y, test_size=0.2, random_state=42)
    X_train_c1, X_test_c1, y_train_s1, y_test_s1 = train_test_split(selected_features_c1, y, test_size=0.2, random_state=42)
    
    scaler=sklearn.preprocessing.StandardScaler()
    X_train_c=scaler.fit_transform(X_train_c)
    X_train_s=scaler.fit_transform(X_train_s)
    X_train_c1=scaler.fit_transform(X_train_c1)

    X_test_c=scaler.fit_transform(X_test_c)
    X_test_c1=scaler.fit_transform(X_test_c1)
    X_test_s=scaler.fit_transform(X_test_s)

    X_train_c=torch.from_numpy(X_train_c.astype(np.float32))
    X_train_c1=torch.from_numpy(X_train_c1.astype(np.float32))
    X_test_c=torch.from_numpy(X_test_c.astype(np.float32))
    X_test_c1=torch.from_numpy(X_test_c1.astype(np.float32))
    y_train_c=torch.from_numpy(y_train_c.astype(np.float32))
    y_test_c=torch.from_numpy(y_test_c.astype(np.float32))
    X_train_s=torch.from_numpy(X_train_s.astype(np.float32))
    X_test_s=torch.from_numpy(X_test_s.astype(np.float32))
    y_train_s=torch.from_numpy(y_train_s.astype(np.float32))
    y_test_s=torch.from_numpy(y_test_s.astype(np.float32))

    y_train_c=y_train_c.view(y_train_c.shape[0],1)
    y_test_c=y_test_c.view(y_test_c.shape[0],1)
    y_train_s=y_train_s.view(y_train_s.shape[0],1)
    y_test_s=y_test_s.view(y_test_s.shape[0],1)
    return X_train_c, X_test_c, y_train_c, y_test_c, X_train_s, X_test_s, y_train_s, y_test_s,X_train_c1, X_test_c1

class Logistic_Reg_model(torch.nn.Module):
   def __init__(self,no_input_features):
      super(Logistic_Reg_model,self).__init__()
      self.layer1=torch.nn.Linear(no_input_features,20)
      self.layer2=torch.nn.Linear(20,1)
   def forward(self,x):
      y_predicted=self.layer1(x)
      y_predicted=torch.sigmoid(self.layer2(y_predicted))
      return y_predicted
 
class bottle_net(torch.nn.Module):
   def __init__(self,no_input_features):
      super(bottle_net,self).__init__()
      self.layer1=torch.nn.Linear(no_input_features,20)
      self.layer2=torch.nn.Linear(20,20)
      self.layer3=torch.nn.Linear(20,2)
      self.layer4=torch.nn.Linear(2,1)
   def forward(self,x):
      y_predicted=self.layer1(x)
      y_predicted=self.layer2(y_predicted)
      y_predicted=self.layer3(y_predicted)
      y_predicted=self.layer4(y_predicted)
      return y_predicted
  
class simulate_net(torch.nn.Module):
   def __init__(self,no_input_features):
      super(simulate_net,self).__init__()
      self.layer1=torch.nn.Linear(no_input_features,20)
      self.layer2=torch.nn.Linear(20,20)
      self.layer3=torch.nn.Linear(20,2)
      self.layer4=torch.nn.Linear(2,1)
   def forward(self,x):
      y_predicted=self.layer1(x)
      y_predicted=self.layer2(y_predicted)
      y_predicted=self.layer3(y_predicted)
      y_predicted=self.layer4(y_predicted)
      return y_predicted
 
class top_net(torch.nn.Module):
   def __init__(self,client_num):
      super(top_net,self).__init__()
      #self.layer1=torch.nn.Linear(no_input_features,20)
      self.layer2=torch.nn.Linear(client_num*2,1)
   def forward(self,x):
      #y_predicted=self.layer1(x)
      y_predicted=torch.sigmoid(x)
      return y_predicted
 

###################################上文无关###################################
criterion=torch.nn.BCELoss()
data = load_breast_cancer()
X = data.data
y = data.target
feature_indices_s = [18,19]
feature_indices_c1 = [1,3,5,7,9,11,13,15,17]
feature_indices_c = [i for i in range(30) if i not in feature_indices_s and i not in feature_indices_c1]
X_c = X[:, feature_indices_c]

scaler1=sklearn.preprocessing.StandardScaler()
all_X_train=scaler1.fit_transform(X_c)
all_X_train=torch.from_numpy(all_X_train.astype(np.float32))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_c)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

labels01 = kmeans.labels_

labels10=deepcopy(labels01)
#默认第一簇标签排序是0,1
kk=(labels01==y)
kkk=kk.sum()
print(kkk/len(y))
labels01=torch.from_numpy(labels01.astype(np.float32))
labels01=labels01.view(labels01.shape[0],1)
#第二簇标签排序是1,0
for xx in range(len(labels10)):
   if labels10[xx]==0:
      labels10[xx]=1
   else:
      labels10[xx]=0
kk=(labels10==y)
kkk=kk.sum()
print(kkk/len(y))
labels10=torch.from_numpy(labels10.astype(np.float32))
labels10=labels10.view(labels10.shape[0],1)

c_features=19
model_c=bottle_net(c_features)
model_c0=bottle_net(9)
model_s=bottle_net(2)

model_c1=deepcopy(model_c)
model_c2=deepcopy(model_c1)


optimizer_c1=torch.optim.SGD(model_c1.parameters(),lr=0.01)
optimizer_c2=torch.optim.SGD(model_c2.parameters(),lr=0.01)
number_of_epochs=2000
for epoch in range(number_of_epochs):
    y_prediction=model_c1(all_X_train)
    y_prediction=torch.sigmoid(y_prediction)
    loss=criterion(y_prediction,labels01)
    optimizer_c1.zero_grad()
    loss.backward()
    optimizer_c1.step()

    if (epoch+1)%10 == 0:
        print('epoch:', epoch+1,',loss=',loss.item())
for epoch in range(number_of_epochs):
    y_prediction=model_c2(all_X_train)
    y_prediction=torch.sigmoid(y_prediction)
    loss=criterion(y_prediction,labels10)
    optimizer_c2.zero_grad()
    loss.backward()
    optimizer_c2.step()
    
    if (epoch+1)%10 == 0:
        print('epoch:', epoch+1,',loss=',loss.item())


###################################下文无关###################################

#X_train_c has 28 features, X_train_s has 2 features
X_train_c, X_test_c, y_train_c, y_test_c, X_train_s, X_test_s, y_train_s, y_test_s,X_train_c1, X_test_c1=load_data()
# 创建逻辑回归模型c
criterion=torch.nn.BCELoss()


c_features=28
#model_c=bottle_net(c_features)
optimizer_c=torch.optim.SGD(model_c.parameters(),lr=0.01)
optimizer_c0=torch.optim.SGD(model_c0.parameters(),lr=0.01)

s_features=2
#model_s=bottle_net(s_features)
optimizer_s=torch.optim.SGD(model_s.parameters(),lr=0.01)

client_n=2

#optimizer_t=torch.optim.SGD(model_t.parameters(),lr=0.01)

number_of_epochs=2000
for epoch in range(number_of_epochs):
    y_prediction1=model_c(X_train_c)
    y_prediction0=model_c0(X_train_c1)
    y_prediction2=model_s(X_train_s)
    #V=torch.cat((y_prediction1,y_prediction2), dim=1)
    V=y_prediction1+y_prediction2+y_prediction0
    #V=y_prediction1
    y_prediction=torch.sigmoid(V)
    loss=criterion(y_prediction,y_train_s)#
    optimizer_c.zero_grad()
    optimizer_c0.zero_grad()
    optimizer_s.zero_grad()
    #optimizer_t.zero_grad()
    loss.backward()
    optimizer_c.step()
    optimizer_c0.step()
    optimizer_s.step()
    #optimizer_t.step()
    if (epoch+1)%10 == 0:
        print('epoch:', epoch+1,',loss=',loss.item())
with torch.no_grad():
    y_prediction1=model_c(X_test_c)
    y_prediction0=model_c0(X_test_c1)
    y_prediction2=model_s(X_test_s)
    #V=torch.cat((y_prediction1,y_prediction2), dim=1)
    V=y_prediction1+y_prediction2+y_prediction0#
    #V=y_prediction1
    y_pred=torch.sigmoid(V)
    
    y_pred_class=y_pred.round()
    accuracy=(y_pred_class.eq(y_test_c).sum())/float(y_test_c.shape[0])#y_test_c=y_test_s
    print("acc: ", accuracy.item())


bottle_layer1 = model_c1.layer1
bottle_layer2 = model_c2.layer1
bottle_layer = model_c.layer1
b1,w1,cc1,c1=compare_layers_cosine_similarity(bottle_layer,bottle_layer1)
b2,w2,cc2,c2=compare_layers_cosine_similarity(bottle_layer,bottle_layer2)
print(f"c1: {cc1}, c2: {cc2}")
'''bottle_layer1 = model_c1.layer2
bottle_layer2 = model_c2.layer2
bottle_layer = model_c.layer2
b3,w3,cc3,c3=compare_layers_cosine_similarity(bottle_layer,bottle_layer1)
b4,w4,cc4,c4=compare_layers_cosine_similarity(bottle_layer,bottle_layer2)
print(f"c1: {b3,w3,cc3,c3}, c2: {b4,w4,cc4,c4}")
#70% attack success
bottle_layer1 = model_c1.layer3
bottle_layer2 = model_c2.layer3
bottle_layer = model_c.layer3
b3,w3,cc3,c3=compare_layers_cosine_similarity(bottle_layer,bottle_layer1)
b4,w4,cc4,c4=compare_layers_cosine_similarity(bottle_layer,bottle_layer2)
print(f"c1: {b3,w3,cc3,c3}, c2: {b4,w4,cc4,c4}")
bottle_layer1 = model_c1.layer4
bottle_layer2 = model_c2.layer4
bottle_layer = model_c.layer4
b3,w3,cc3,c3=compare_layers_cosine_similarity(bottle_layer,bottle_layer1)
b4,w4,cc4,c4=compare_layers_cosine_similarity(bottle_layer,bottle_layer2)
print(f"c1: {b3,w3,cc3,c3}, c2: {b4,w4,cc4,c4}")'''

if cc1>cc2:
   print("Get attack Model: model_1.")
   
   attack_model=model_c1
else:
   print("Get attack Model: model_2.")
   
   attack_model=model_c2
# 创建逻辑回归模型s

with torch.no_grad():
    y_pred=attack_model(X_test_c)
    y_pred=torch.sigmoid(y_pred)
    y_pred_class=y_pred.round()
    accuracy=(y_pred_class.eq(y_test_c).sum())/float(y_test_c.shape[0])
    print("Attack acc: ", accuracy.item())
