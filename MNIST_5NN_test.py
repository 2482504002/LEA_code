from torchvision import datasets, transforms
import torch.utils.data as Data
import numpy as np
from copy import deepcopy
import pickle
import torch.nn as nn
import torch.nn.functional as F
class Layer:
    lamda = 3  # 正则化惩罚系数
    w=0
    b=0
    last_node_num=0
    node_num=0
    batch_size=0
    activation=''
    learning_rate=0.1
    x=0
    activation_data =0
    def __init__(self,last_node_num,node_num,batch_size,activation):
        self.last_node_num=last_node_num
        self.node_num=node_num
        self.w = np.random.normal(scale=0.01, size=(last_node_num,node_num))  # 生成随机正太分布的w矩阵
        self.b = np.zeros((batch_size, node_num))
        self.activation=activation
        self.batch_size=batch_size
 
    def forward(self,data):
        self.x=data
        data=np.dot(data, self.w) + self.b
        if self.activation=="Sigmoid":
            data=1 / (1 + np.exp(-data))
            # print(data.mean())
        if self.activation=="Tahn":
            data = (np.exp(data)- np.exp(-data)) / (np.exp(data)+ np.exp(-data))
        if self.activation == "Relu":
            data= (np.abs(data)+data)/2.0
        self.activation_data = data
        return data

    def backward(self,y):
        if self.activation == "Sigmoid":
            y = self.activation_data * (1 - self.activation_data) * y
        if self.activation == "Tahn":
            y = (1 - self.activation_data**2) * y
        if self.activation=="Relu":
            self.activation_data[self.activation_data <= 0] = 0
            self.activation_data[self.activation_data > 0] = 1
            y = self.activation_data *y
        w_gradient=np.dot(self.x.T, y)
        b_gradient=y
        y=np.dot(y,self.w.T)
        self.w = self.w - (w_gradient+(self.lamda*self.w)) / self.batch_size * self.learning_rate
        self.b = self.b - b_gradient / self.batch_size * self.learning_rate
        return y
 
class Softmax (Layer):
    y_hat=[]
    def __init__(self,node_num):
        self.node_num=node_num
        pass
    def forward(self,data):
        data = np.exp(data.T)  # 先把每个元素都进行exp运算
        # print(label)
        sum = data.sum(axis=0)  # 对于每一行进行求和操作
        # print((label/sum).T.sum(axis=1))
        self.y_hat=(data / sum).T
        return self.y_hat  # 通过广播机制，使每行分别除以各种的和
    def backward(self,y):
        return self.y_hat-y
    
class Net:
    def __init__(self,batch_size,input_num):
        self.batch_size=batch_size#256
        self.input_num=input_num#784
        self.layers=[]
        self.batch_size=batch_size
        self.input_num=input_num
        pass
    def add(self,layer_type,node_num,activation=""):
        if (len(self.layers)==0):
            last_node_num=self.input_num
        else:
            last_node_num=self.layers[-1].node_num #获取上一层的节点个数
        if (layer_type=='Softmax'):
            self.layers.append(Softmax((node_num)))
        else:
            self.layers.append(Layer(last_node_num,node_num,self.batch_size,activation))
    def forward(self,data):
        for layer in self.layers:
            data=layer.forward(data)
        return data #返回最后输出的data用于反向传播
    def harf_forward(self,data):
        for i in range(3):
            data=self.layers[i].forward(data)
        return data
    def backward(self,y_hat):
        dydx=y_hat
        for layer in reversed(self.layers):
            dydx=layer.backward(dydx)

class Simu_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5) # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = nn.Conv2d(10, 20, 3) # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20*10*10, 500) # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(500, 10) # 输入通道数是500，输出通道数是10，即10分类
    def forward(self,x):
        in_size = x.size(0) # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
        out = self.conv1(x) # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = F.relu(out) # batch*10*24*24（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2) # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = self.conv2(out) # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out) # batch*20*10*10
        out = out.view(in_size, -1) # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = self.fc1(out) # batch*2000 -> batch*500
        out = F.relu(out) # batch*500
        out = self.fc2(out) # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1) # 计算log(softmax(x))
        return out

batch_size = 300
epoch=20
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


with open('MNIST_attack_model01.pkl', 'rb') as f:
    model01 = pickle.load(f)
with open('MNIST_attack_model23.pkl', 'rb') as f:
    model23 = pickle.load(f)
with open('MNIST_attack_model45.pkl', 'rb') as f:
    model45 = pickle.load(f)
with open('MNIST_attack_model67.pkl', 'rb') as f:
    model67 = pickle.load(f)
with open('MNIST_attack_model89.pkl', 'rb') as f:
    model89 = pickle.load(f)
all_acc=[]
for batch_idx, (data, target) in enumerate(test_loader):
    if (batch_idx>10):
        break
    data = np.squeeze(data.numpy()).reshape(batch_size, 784)  # 把张量中维度为1的维度去掉,并且改变维度为(64,784)
    target = target.numpy()  # x矩阵 (64,10)
    new_net1=deepcopy(model01)
    new_net2=deepcopy(model23)
    new_net3=deepcopy(model45)
    new_net4=deepcopy(model67)
    new_net5=deepcopy(model89)
    y_hat1 = new_net1.forward(data)
    y_hat2 = new_net2.forward(data)
    y_hat3 = new_net3.forward(data)
    y_hat4 = new_net4.forward(data)
    y_hat5 = new_net5.forward(data)
    all_hat=[y_hat1,y_hat2,y_hat3,y_hat4,y_hat5]
    
    simu_label=[]
    for idx in range(batch_size):
        max_id=[]
        for hat in all_hat:
            max_id.append([max(hat[idx]), np.argmax(hat[idx])])
        simu_label.append(max_id[max_id.index(max(max_id))][1])
    simu_label=np.array(simu_label)
    acc = simu_label == target
    acc = acc + 0
    accc=acc.mean()
    print(accc)
    a=1
