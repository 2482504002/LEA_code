from torchvision import datasets, transforms
import torch.utils.data as Data
import torch
import numpy as np
from copy import deepcopy
from MNIST_3kmeans import *
import random
import torch.nn.functional as F
import time

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

def generated_nets(batch_size=300):
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

def compare_matrices(matrix1, matrix2):
    #matrix1 (torch.Tensor): 形状为 [20, 5] 的第一个矩阵。
    #matrix2 (torch.Tensor): 形状为 [20, 5] 的第二个矩阵。
    frobenius_diff=0
    cos_sim=0
    for i in range(3):
        frobenius_diff += torch.norm(torch.from_numpy(matrix1[i]) - torch.from_numpy(matrix2[i]), p='fro')
        cos_sim += F.cosine_similarity(torch.from_numpy(matrix1[i]).view(-1), torch.from_numpy(matrix2[i]).view(-1), dim=0)
    return (cos_sim, frobenius_diff.item())



start = time.time()
batch_size = 300
epoch=10
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
    print(f"VFL Epoch: {e}, Train acc: {accc:.4f}")
    
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
    
bottle_layer_c=net_c.layers[:3]
for la in range(3):
    bottle_layer_c[la]=np.vstack((bottle_layer_c[la].b, bottle_layer_c[la].w))



batch_size = 300
epoch=10
m=60000

k_data, k_labels, test_loader= load_data()

kk_data, kk_labels=[], []
class_num=4
group_num=class_num//2
l=0
while(l<class_num):
    b1=np.where(k_labels==l)
    kk_data1=np.array([k_data[idx] for idx in b1[0]])
    kk_labels1=np.array([k_labels[idx] for idx in b1[0]])
    l+=1
    b2=np.where(k_labels==l)
    kk_data2=np.array([k_data[idx] for idx in b2[0]])
    kk_labels2=np.array([k_labels[idx] for idx in b2[0]])
    kk_data.append(np.vstack((kk_data1, kk_data2)))
    kk_labels.append(np.hstack((kk_labels1, kk_labels2)))
    l+=1

all_simulate_labels=[]
all_nets=[]
l=0

class_num1=0
while(l<class_num):   
    simulate_labels_list=[]
    nets=[]
    len1=len(np.where(kk_labels[l//2]==l)[0])
    len2=len(np.where(kk_labels[l//2]==(l+1))[0])
    for i in range(class_num-1,class_num1-1,-1):
        for j in range(class_num-1,class_num1-1,-1):
            if i==j:
                continue
            simulate_labels = np.hstack((np.full(len1, i), np.full(len2, j)))
            simulate_labels_list.append(simulate_labels)
            simulate_net=Net(batch_size,784)
            simulate_net.layers=net_c.layers+net_s.layers
            nets.append(simulate_net)
    all_simulate_labels.append(simulate_labels_list)
    all_nets.append(nets)
    l+=2
    class_num1+=2
    
all_bottle_layers=[]
for group in range(group_num):#0,8
    bottle_layers=[]
    k_data=kk_data[group]
    label_num=len(all_simulate_labels[group][0])
    prob_num=len(all_simulate_labels[group])
    nets=all_nets[group]
    for p in range(prob_num):
        k_labels=all_simulate_labels[group][prob_num-p-1]
        simulate_net=nets[p]
        for e in range(epoch):
            for i in range(label_num//batch_size):
                data,target = get_random_data_targets(k_data, k_labels, batch_size, label_num)
                
                y_hat = simulate_net.forward(data)
                simulate_net.backward( np.eye(10)[target] )
                acc = y_hat.argmax(axis=1) == target
                acc = acc + 0  
                accc = acc.mean()  
            print(f"Group: {group}, Net: {p}, Epoch: {e}, Train acc: {accc:.4f}")
        all_nets[group][p]=simulate_net
        bottle_layers.append(simulate_net.layers[:3])
        for la in range(3):
            bottle_layers[p][la]=np.vstack((bottle_layers[p][la].b, bottle_layers[p][la].w))
    all_bottle_layers.append(bottle_layers)

for group in range(group_num):
    c_l=[]
    for x in range(len(all_bottle_layers[group])):
        c=compare_matrices(bottle_layer_c,all_bottle_layers[group][x])
        c_l.append([c,group,x])

    attack_idx=c_l.index(max(c_l))
    c,group1,x1=c_l[attack_idx][0],c_l[attack_idx][1],c_l[attack_idx][2]
    print(f"Get attack model:{all_simulate_labels[group1][len(all_bottle_layers[group1])-x1-1][0]}{all_simulate_labels[group1][len(all_bottle_layers[group1])-x1-1][-1]}")
    attack_net=all_nets[group1][x1]
    with open(f'MNIST_attack_model{all_simulate_labels[group1][len(all_bottle_layers[group1])-x1-1][0]}{all_simulate_labels[group1][len(all_bottle_layers[group1])-x1-1][-1]}.pkl', 'wb') as file:
        pickle.dump(attack_net, file)

end = time.time()
print('总用时：'+str(end-start)+'秒！')

