from torchvision import datasets, transforms
import torch.utils.data as Data
import numpy as np
from copy import deepcopy
from MNIST_3kmeans import *
import random

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
    def backward(self,y_hat):
        dydx=y_hat
        for layer in reversed(self.layers):
            dydx=layer.backward(dydx)
        return dydx

def load_data():
    with open('MNIST_label.pkl', 'rb') as f:
        k_labels = pickle.load(f).astype(np.int64) 
    with open('MNIST_data.pkl', 'rb') as f2:
        k_data = pickle.load(f2).astype(np.float32)
    k_data = k_data/255.0
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return k_data, k_labels, test_loader

def generated_nets():
    net_c = Net(batch_size,784)
    net_c.add("", 256, activation="Relu")
    net_c.add("", 64, activation="Relu")
    net_c.add("", 10, activation="Relu")

    net_s = Net(10,10)
    net_s.add("Softmax", 10)
    return net_c, net_s
    
def get_random_data_targets(k_data, k_labels, batch_size, m):
    k_data_list, k_labels_list=[], []
    selected_numbers = random.sample([x for x in range(m)], 300)
    for i in selected_numbers:
        k_data_list.append(k_data[i])
        k_labels_list.append(k_labels[i])
    data = np.stack(k_data_list)
    data = np.squeeze(data).reshape(batch_size, 784)
    target=np.array(k_labels_list)
    return data,target

batch_size = 300
epoch=50
m=60000
net_c, net_s=generated_nets()
k_data, k_labels, test_loader= load_data()

for e in range(epoch):
    for i in range(m//batch_size):
        data,target = get_random_data_targets(k_data, k_labels, batch_size, m)
        
        y_hat = net_c.forward(data)
        y_hat = net_s.forward(y_hat)
        dydx = net_s.backward( np.eye(10)[target] )
        net_c.backward(dydx)

        acc = y_hat.argmax(axis=1) == target
        acc = acc + 0  
        accc = acc.mean()  
    print(f"Epoch: {e}   Train acc: {accc:.4f}")
    
    all_acc = []
    for batch_idx, (data, target) in enumerate(test_loader):
        if (batch_idx>10):
            break
        data = np.squeeze(data.numpy()).reshape(batch_size, 784)  
        target = target.numpy() 
        new_net1=deepcopy(net_c)
        new_net2=deepcopy(net_s)
        y_hat = new_net1.forward(data)
        y_hat = new_net2.forward(y_hat)
        y_t=y_hat.argmax(axis=1)
        acc = y_t == target
        acc = acc + 0
        all_acc.append(acc.mean())
    print(f"Test acc: {(sum(all_acc) / len(all_acc)):.4f}") 
    
    