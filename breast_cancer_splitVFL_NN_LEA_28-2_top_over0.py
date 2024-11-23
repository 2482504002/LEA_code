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
   weight1_flattened = weight1.view(weight1.numel())
   weight2_flattened = weight2.view(weight2.numel())
   l1=torch.sum(weight1_flattened-weight2_flattened)
   weight_similarity = cosine_similarity(weight1_flattened, weight2_flattened,dim=0)

   if layer1.bias is not None and layer2.bias is not None:
      
      b_layer1=torch.cat(((layer1.bias.data).unsqueeze(1),layer1.weight.data ), dim=1)
      b_layer2=torch.cat(((layer2.bias.data).unsqueeze(1),layer2.weight.data ), dim=1)
      l1=(torch.sum(b_layer1-b_layer2))
      cos_sim = cosine_similarity(b_layer1.view(-1), b_layer2.view(-1), dim=0)
      # 将权重展平为一维向量

      bias_similarity = cosine_similarity(layer1.bias.data, layer2.bias.data,dim=0 )
   else:
      bias_similarity,cos_sim=0,0
   return bias_similarity,weight_similarity,cos_sim,(weight_similarity + bias_similarity) / 2,l1
 
def load_data():
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_indices_s = [18,19]
    feature_indices_c = [i for i in range(30) if i not in feature_indices_s]
    selected_features_c = X[:, feature_indices_c]
    selected_features_s = X[:, feature_indices_s]
    # 将数据集分为训练集和测试集
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(selected_features_c, y, test_size=0.2, random_state=42)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(selected_features_s, y, test_size=0.2, random_state=42)
    
    scaler=sklearn.preprocessing.StandardScaler()
    X_train_c=scaler.fit_transform(X_train_c)
    X_train_s=scaler.fit_transform(X_train_s)
    X_test_c=scaler.fit_transform(X_test_c)
    X_test_s=scaler.fit_transform(X_test_s)

    X_train_c=torch.from_numpy(X_train_c.astype(np.float32))
    X_test_c=torch.from_numpy(X_test_c.astype(np.float32))
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
    return X_train_c, X_test_c, y_train_c, y_test_c, X_train_s, X_test_s, y_train_s, y_test_s
 
class bottle_net(torch.nn.Module):
   def __init__(self,no_input_features):
      super(bottle_net,self).__init__()
      self.layer1=torch.nn.Linear(no_input_features,20)
   def forward(self,x):
      y_predicted=torch.relu(self.layer1(x))
      return y_predicted
  
class simulate_net(torch.nn.Module):
   def __init__(self,no_input_features):
      super(simulate_net,self).__init__()
      self.layer1=torch.nn.Linear(no_input_features,20)
      self.layer2=torch.nn.Linear(20,1)
      self.layer2.weight.data=torch.clamp(self.layer2.weight.data, 0,1)
      
   def forward(self,x):
      y_predicted=torch.relu(self.layer1(x))
      y_predicted=torch.sigmoid(self.layer2(y_predicted))
      return y_predicted
 
class top_net(torch.nn.Module):
   def __init__(self,client_num):
      super(top_net,self).__init__()
      self.layer2=torch.nn.Linear(client_num*20,1)
      self.layer2.weight.data=torch.clamp(self.layer2.weight.data, 0,1)
      #self.layer2.weight.data[:,:client_num*10]=torch.clamp(self.layer2.weight.data[:,:client_num*10],0,1)
      #self.layer2.weight.data[:,client_num*10:]=torch.clamp(self.layer2.weight.data[:,client_num*10:],-1,0)
      
      #self.layer2.weight.data[client_num*10:]=torch.clamp(self.layer2.weight.data, -1,0)

   def forward(self,x):
      y_predicted=torch.sigmoid(self.layer2(x))
      return y_predicted


###################################上文无关###################################
X_train_c, X_test_c, y_train_c, y_test_c, X_train_s, X_test_s, y_train_s, y_test_s=load_data()
feature_indices_c = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,28,29]
y_train_s=torch.from_numpy(y_train_s.astype(np.float32))
y_train_s=y_train_s.view(y_train_s.shape[0],1)
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

c_features=28
model_c=bottle_net(c_features)
model_s=bottle_net(2)
model_t=top_net(2)


model_c1=simulate_net(c_features)
model_c1.layer1=deepcopy(model_c.layer1)
#model_c1.layer2.weight.data=torch.clamp(model_c1.layer2.weight.data, 0, 1)
model_c2=deepcopy(model_c1)

number_of_epochs=1000
learning_rate=0.01

for epoch in range(number_of_epochs):
   y_prediction=model_c1.layer1(X_train_c)
   y_h=torch.sigmoid(model_c1.layer2(y_prediction))
   yy=y_h.detach().numpy().reshape(455)
   dy=labels01-y_h

   cost = -np.sum(y_train_c * np.log(yy) + (1 - y_train_c) * np.log(1 - yy)) / len(X_train_c)
   # 计算梯度
   tmp_xa=torch.matmul(X_train_c.T,dy)
   tmp_xt=torch.matmul(y_prediction.T,dy)
   tmp_xaa=torch.matmul(tmp_xa,model_c1.layer2.weight.data)

   l_a=(tmp_xaa/(-len(X_train_c))).reshape(28,20)
   l_t=(tmp_xt/(-len(X_train_c))).reshape(20,1)
   if epoch == 0:
         deta1=l_a.T
   
   model_c1.layer1.weight.data = model_c1.layer1.weight.data - learning_rate * l_a.T
   model_c1.layer2.weight.data = model_c1.layer2.weight.data - learning_rate * l_t.T

   if (epoch+1)%200 == 0:
        print('epoch:', epoch+1,',loss=',cost)
        
for epoch in range(number_of_epochs):
   y_prediction=model_c2.layer1(X_train_c)
   y_h=torch.sigmoid(model_c2.layer2(y_prediction))
   yy=y_h.detach().numpy().reshape(455)
   dy=labels10-y_h
   epsilon = 1e-18
   clipped_y_h = np.clip(yy, epsilon, 1 - epsilon)
   cost = -np.sum(y_train_c * np.log(clipped_y_h) + (1 - y_train_c) * np.log(1 - clipped_y_h)) / len(X_train_c)
   # 计算梯度
   tmp_xa=torch.matmul(X_train_c.T,dy)
   tmp_xt=torch.matmul(y_prediction.T,dy)
   tmp_xaa=torch.matmul(tmp_xa,model_c2.layer2.weight.data)

   l_a=(tmp_xaa/(-len(X_train_c))).reshape(28,20)
   l_t=(tmp_xt/(-len(X_train_c))).reshape(20,1)
   if epoch == 0:
         deta2=l_a.T
   model_c2.layer1.weight.data = model_c2.layer1.weight.data - learning_rate * l_a.T
   model_c2.layer2.weight.data = model_c2.layer2.weight.data - learning_rate * l_t.T
    
   if (epoch+1)%200 == 0:
        print('epoch:', epoch+1,',loss=',cost)

    
with torch.no_grad():
    y_prediction1=model_c1(X_test_c)
    y_prediction2=model_c2(X_test_c)
    
    y_pred_class1=y_prediction1.round().reshape(114)
    y_pred_class1=np.array(y_pred_class1)
 
    y_pred_class2=y_prediction2.round().reshape(114)
    y_pred_class2=np.array(y_pred_class2)

    accuracy1=(y_test_s==y_pred_class1).sum()/len(y_test_s)
    accuracy2=(y_test_s==y_pred_class2).sum()/len(y_test_s)
    
    print("model_c1 acc: ", accuracy1.item())
    print("model_c2 acc: ", accuracy2.item())

for epoch in range(number_of_epochs):
   y_prediction1=model_c(X_train_c)
   y_prediction2=model_s(X_train_s)
   V=torch.cat((y_prediction1,y_prediction2), dim=1)
   y_prediction =model_t(V)
   yy=y_prediction.detach().numpy().reshape(455)
   dy=y_train_s-y_prediction
   
   epsilon = 1e-18
   clipped_y_h = np.clip(yy, epsilon, 1 - epsilon)
   
   cost = -np.sum(y_train_c * np.log(clipped_y_h) + (1 - y_train_c) * np.log(1 - clipped_y_h)) / len(X_train_c)
   
   # 计算梯度
   #tmp_xa=torch.matmul(X_train_c.T,dy)
   tmp_xa=torch.matmul(dy,model_t.layer2.weight.data[:,:20])
   #tmp_xb=torch.matmul(X_train_s.T,dy)
   tmp_xb=torch.matmul(dy,model_t.layer2.weight.data[:,20:])
   tmp_xt=torch.matmul(V.T,dy)
   tmp_xaa=torch.matmul(X_train_c.T,tmp_xa)
   tmp_xbb=torch.matmul(X_train_s.T,tmp_xb)

   l_a=(tmp_xaa/(-len(X_train_c))).reshape(28,20)
   l_b=(tmp_xbb/(-len(X_train_s))).reshape(2,20)
   l_t=(tmp_xt/(-len(X_train_s))).reshape(40,1)
   if epoch == 0:
         detac=l_a.T
   
   model_c.layer1.weight.data = model_c.layer1.weight.data - learning_rate * l_a.T
   model_s.layer1.weight.data = model_s.layer1.weight.data - learning_rate * l_b.T
   model_t.layer2.weight.data = model_t.layer2.weight.data - learning_rate * l_t.T

   if (epoch+1)%200 == 0:
        print('epoch:', epoch+1,',loss=',cost)
        
with torch.no_grad():
    y_prediction1=model_c(X_test_c)
    y_prediction2=model_s(X_test_s)
    V=torch.cat((y_prediction1,y_prediction2), dim=1)
    y_pred =model_t(V)
    
    y_pred_class=y_pred.round().reshape(114)
    y_pred_class=np.array(y_pred_class)

    accuracy=(y_pred_class==y_test_s).sum()/len(y_pred_class)
    print("model_c acc: ", accuracy.item())
    
bottle_layer1 = model_c1.layer1
bottle_layer2 = model_c2.layer1
bottle_layer = model_c.layer1
b1,w1,cc1,c1,l1=compare_layers_cosine_similarity(bottle_layer,bottle_layer1)
b2,w2,cc2,c2,l2=compare_layers_cosine_similarity(bottle_layer,bottle_layer2)
b3,w3,cc3,c3,l3=compare_layers_cosine_similarity(bottle_layer1,bottle_layer2)
print(f"c1,l1: {c1,l1}")
print(f"c2,l2: {c2,l2}")
print(f"c3,l3: {c3,l3}")

c1=cosine_similarity(detac, deta1,dim=0).sum()/len(detac)
c2=cosine_similarity(detac, deta2,dim=0).sum()/len(detac)
   
print("deta_c1: ",c1)
print("deta_c2: ",c2)

'''
c3=compare_layers_cosine_similarity1(bottle_layer1,bottle_layer2)
print(f"c3: {c3}")
c1=compare_layers_cosine_similarity1(bottle_layer,bottle_layer1)
c2=compare_layers_cosine_similarity1(bottle_layer,bottle_layer2)

print(f"c1: {c1}")
print(f"c2: {c2}")
'''
'''bottle_layer1 = model_c1.layer0
bottle_layer2 = model_c2.layer0
bottle_layer = model_c.layer0
b1,w1,cc1,c1=compare_layers_cosine_similarity(bottle_layer,bottle_layer1)
b2,w2,cc2,c2=compare_layers_cosine_similarity(bottle_layer,bottle_layer2)
b2,w2,cc3,c2=compare_layers_cosine_similarity(bottle_layer1,bottle_layer2)
print(f"c1: {cc1}")
print(f"c2: {cc2}")
print(f"c3: {cc3}")

bottle_layer1 = model_c1.layer00
bottle_layer2 = model_c2.layer00
bottle_layer = model_c.layer00
b1,w1,cc1,c1=compare_layers_cosine_similarity(bottle_layer,bottle_layer1)
b2,w2,cc2,c2=compare_layers_cosine_similarity(bottle_layer,bottle_layer2)
b2,w2,cc3,c2=compare_layers_cosine_similarity(bottle_layer1,bottle_layer2)
print(f"c1: {cc1}")
print(f"c2: {cc2}")
print(f"c3: {cc3}")'''
deta_sum=detac.sum()
deta_sum1=deta1.sum()
deta_sum2=deta2.sum()
print(deta_sum,deta_sum1,deta_sum2)

'''if deta_sum>0:
   if c1>c2:
      print("Get attack Model: model_1.")
      attack_model=model_c1
   else:
      print("Get attack Model: model_2.")
      attack_model=model_c2
if deta_sum<0:
   if c1>c2:
      print("Get attack Model: model_1.")
      attack_model=model_c2
   else:
      print("Get attack Model: model_2.")
      attack_model=model_c1
'''
if c1>c2:
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
