from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import torch

def load_data():
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_indices_c = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,28,29]
    feature_indices_s = [18,19]
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
    return X_train_c, X_test_c, y_train_c, y_test_c, X_train_s, X_test_s, y_train_s, y_test_s

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
     #self.layer2=torch.nn.Linear(20,1)
  def forward(self,x):
     y_predicted=self.layer1(x)
     #y_predicted=torch.sigmoid(self.layer2(y_predicted))
     return y_predicted
 
class top_net(torch.nn.Module):
  def __init__(self):
     super(top_net,self).__init__()
     #self.layer1=torch.nn.Linear(no_input_features,20)
     self.layer2=torch.nn.Linear(40,1)
  def forward(self,x):
     #y_predicted=self.layer1(x)
     y_predicted=torch.sigmoid(self.layer2(x))
     return y_predicted
 
#X_train_c has 28 features, X_train_s has 2 features
X_train_c, X_test_c, y_train_c, y_test_c, X_train_s, X_test_s, y_train_s, y_test_s=load_data()
# 创建逻辑回归模型c
criterion=torch.nn.BCELoss()

c_features=28
model_c=bottle_net(c_features)
optimizer_c=torch.optim.SGD(model_c.parameters(),lr=0.01)

s_features=2
model_s=bottle_net(s_features)
optimizer_s=torch.optim.SGD(model_s.parameters(),lr=0.01)

model_t=top_net()
optimizer_t=torch.optim.SGD(model_t.parameters(),lr=0.01)

number_of_epochs=2000
for epoch in range(number_of_epochs):
    y_prediction1=model_c(X_train_c)
    y_prediction2=model_s(X_train_s)
    V=torch.cat((y_prediction1,y_prediction2), dim=1)
    
    y_prediction =model_t(V)
    loss=criterion(y_prediction,y_train_s)#
    optimizer_c.zero_grad()
    optimizer_s.zero_grad()
    optimizer_t.zero_grad()
    loss.backward()
    optimizer_c.step()
    optimizer_s.step()
    optimizer_t.step()
    if (epoch+1)%10 == 0:
        print('epoch:', epoch+1,',loss=',loss.item())
with torch.no_grad():
    y_prediction1=model_c(X_test_c)
    y_prediction2=model_s(X_test_s)
    V=torch.cat((y_prediction1,y_prediction2), dim=1)
    y_pred =model_t(V)
    
    y_pred_class=y_pred.round()
    accuracy=(y_pred_class.eq(y_test_c).sum())/float(y_test_c.shape[0])#y_test_c=y_test_s
    print("acc: ", accuracy.item())


# 创建逻辑回归模型s

