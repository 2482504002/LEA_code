from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import torch
from copy import deepcopy

def load_data():
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler=sklearn.preprocessing.StandardScaler()
    X_train_c=scaler.fit_transform(X_train_c)
    X_test_c=scaler.fit_transform(X_test_c)

    X_train_c=torch.from_numpy(X_train_c.astype(np.float32))
    X_test_c=torch.from_numpy(X_test_c.astype(np.float32))
    y_train_c=torch.from_numpy(y_train_c.astype(np.float32))
    y_test_c=torch.from_numpy(y_test_c.astype(np.float32))

    y_train_c=y_train_c.view(y_train_c.shape[0],1)
    y_test_c=y_test_c.view(y_test_c.shape[0],1)

    return X_train_c, X_test_c, y_train_c, y_test_c

class Logistic_Reg_model(torch.nn.Module):
  def __init__(self,no_input_features):
     super(Logistic_Reg_model,self).__init__()
     self.layer1=torch.nn.Linear(no_input_features,20)
     self.layer4=torch.nn.Linear(20,1)
     
  def forward(self,x):
     y_predicted=self.layer1(x)
     y_predicted=torch.sigmoid(self.layer4(y_predicted))
     return y_predicted


X_train, X_test, y_train, y_test=load_data()
# 创建逻辑回归模型
n_features=30
model=Logistic_Reg_model(n_features)
model1=Logistic_Reg_model(n_features)
model2=Logistic_Reg_model(n_features)
model3=Logistic_Reg_model(n_features)
model4=Logistic_Reg_model(n_features)
model5=Logistic_Reg_model(n_features)
'''model=Logistic_Reg_model(n_features)
model1=deepcopy(model)
model2=deepcopy(model)
model3=deepcopy(model)
model4=deepcopy(model)'''

'''model=Logistic_Reg_model(n_features)
model1=Logistic_Reg_model(n_features)
model2=Logistic_Reg_model(n_features)
model3=Logistic_Reg_model(n_features)
model4=Logistic_Reg_model(n_features)
model1.layer1=deepcopy(model.layer1)
model2.layer1=deepcopy(model.layer1)
model3.layer1=deepcopy(model.layer1)
model4.layer1=deepcopy(model.layer1)'''

lr=0.001
criterion=torch.nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
optimizer1=torch.optim.Adam(model1.parameters(),lr=lr)
optimizer2=torch.optim.Adam(model2.parameters(),lr=lr)
optimizer3=torch.optim.Adam(model3.parameters(),lr=lr)
optimizer4=torch.optim.Adam(model4.parameters(),lr=lr)
optimizer5=torch.optim.Adam(model5.parameters(),lr=lr)

number_of_epochs=1000
for epoch in range(number_of_epochs):
    y_prediction=model(X_train)
    loss=criterion(y_prediction,y_train)
    loss.backward()
    if epoch==0:
        detac0=model.layer1.weight.grad.data
    
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1)%10 == 0:
        print('epoch:', epoch+1,',loss=',loss.item())
with torch.no_grad():
    y_pred=model(X_test)
    y_pred_class=y_pred.round()
    accuracy1=(y_pred_class.eq(y_test).sum())/float(y_test.shape[0])
    
    
for epoch in range(number_of_epochs):
    y_prediction=model1(X_train)
    loss=criterion(y_prediction,y_train)
    loss.backward()
    if epoch==0:
        detac1=model1.layer1.weight.grad.data
    optimizer1.step()
    optimizer1.zero_grad()
    if (epoch+1)%10 == 0:
        print('epoch:', epoch+1,',loss=',loss.item())
with torch.no_grad():
    y_pred=model1(X_test)
    y_pred_class=y_pred.round()
    accuracy2=(y_pred_class.eq(y_test).sum())/float(y_test.shape[0])
    

for epoch in range(number_of_epochs):
    y_prediction=model2(X_train)
    loss=criterion(y_prediction,y_train)
    loss.backward()
    if epoch==0:
        detac2=model2.layer1.weight.grad.data
    optimizer2.step()
    optimizer2.zero_grad()
    if (epoch+1)%10 == 0:
        print('epoch:', epoch+1,',loss=',loss.item())
with torch.no_grad():
    y_pred=model2(X_test)
    y_pred_class=y_pred.round()
    accuracy3=(y_pred_class.eq(y_test).sum())/float(y_test.shape[0])
    

for epoch in range(number_of_epochs):
    y_prediction=model3(X_train)
    loss=criterion(y_prediction,y_train)
    loss.backward()
    if epoch==0:
        detac3=model3.layer1.weight.grad.data
    optimizer3.step()
    optimizer3.zero_grad()
    if (epoch+1)%10 == 0:
        print('epoch:', epoch+1,',loss=',loss.item())
with torch.no_grad():
    y_pred=model3(X_test)
    y_pred_class=y_pred.round()
    accuracy4=(y_pred_class.eq(y_test).sum())/float(y_test.shape[0])
    

for epoch in range(number_of_epochs):
    y_prediction=model4(X_train)
    loss=criterion(y_prediction,y_train)
    loss.backward()
    if epoch==0:
        detac4=model4.layer1.weight.grad.data
    optimizer4.step()
    optimizer4.zero_grad()
    if (epoch+1)%10 == 0:
        print('epoch:', epoch+1,',loss=',loss.item())
with torch.no_grad():
    y_pred=model4(X_test)
    y_pred_class=y_pred.round()
    accuracy5=(y_pred_class.eq(y_test).sum())/float(y_test.shape[0])
  
  
y_train1=deepcopy(y_train)
for x in range(len(y_train1)):
    if y_train1[x]==0:
        y_train1[x]=1
    else:
        y_train1[x]=0
for epoch in range(number_of_epochs):
    y_prediction=model5(X_train)
    loss=criterion(y_prediction,y_train1)
    loss.backward()
    if epoch==0:
        detac5=model5.layer1.weight.grad.data
    optimizer5.step()
    optimizer5.zero_grad()
    if (epoch+1)%10 == 0:
        print('epoch:', epoch+1,',loss=',loss.item())
with torch.no_grad():
    y_pred=model5(X_test)
    y_pred_class=y_pred.round()
    accuracy6=(y_pred_class.eq(y_test).sum())/float(y_test.shape[0])


print("acc0: ", accuracy1.item())
print("acc1: ", accuracy2.item())
print("acc2: ", accuracy3.item())
print("acc3: ", accuracy4.item())
print("acc4: ", accuracy5.item())
print("acc5: ", accuracy6.item())

b_l=model.layer1
b_l1=model1.layer1
b_l2=model2.layer1
b_l3=model3.layer1
b_l4=model4.layer1
b_l5=model5.layer1

from torch.nn.functional import cosine_similarity
cc1=cosine_similarity(detac0,detac1,dim=0)
cc2=cosine_similarity(detac0,detac2,dim=0)
cc3=cosine_similarity(detac0,detac3,dim=0)
cc4=cosine_similarity(detac0,detac4,dim=0)
cc5=cosine_similarity(detac0,detac5,dim=0)
cc11=torch.clamp(cc1,0,1)
cc22=torch.clamp(cc2,0,1)
cc33=torch.clamp(cc3,0,1)
cc44=torch.clamp(cc4,0,1)
cc55=torch.clamp(cc5,0,1)
print("grad01: ",cc11)
print("grad02: ",cc22)
print("grad03: ",cc33)
print("grad04: ",cc44)
print("grad05: ",cc55)

detac0=detac0.view(detac0.numel())
detac1=detac1.view(detac1.numel())
detac2=detac2.view(detac2.numel())
detac3=detac3.view(detac3.numel())
detac4=detac4.view(detac4.numel())
detac5=detac5.view(detac5.numel())
detac0=torch.clamp(detac0,0,1)
detac1=torch.clamp(detac1,0,1)
detac2=torch.clamp(detac2,0,1)
detac3=torch.clamp(detac3,0,1)
detac4=torch.clamp(detac4,0,1)
detac5=torch.clamp(detac5,0,1)
cc1=cosine_similarity(detac0,detac1,dim=0)
cc2=cosine_similarity(detac0,detac2,dim=0)
cc3=cosine_similarity(detac0,detac3,dim=0)
cc4=cosine_similarity(detac0,detac4,dim=0)
cc5=cosine_similarity(detac0,detac5,dim=0)
print("grad01: ",cc1)
print("grad02: ",cc2)
print("grad03: ",cc3)
print("grad04: ",cc4)
print("grad05: ",cc5)

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
c1=compare_layers_cosine_similarity(b_l,b_l1)
c2=compare_layers_cosine_similarity(b_l,b_l2)
c3=compare_layers_cosine_similarity(b_l,b_l3)
c4=compare_layers_cosine_similarity(b_l,b_l4)
c5=compare_layers_cosine_similarity(b_l,b_l5)



print("model01: ",c1[2])
print("model02: ",c2[2])
print("model03: ",c3[2])
print("model04: ",c4[2])
print("model05: ",c5[2])