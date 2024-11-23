from torchvision import datasets, transforms
import torch.utils.data as Data
import numpy as np
from copy import deepcopy
import pickle
import random
import torch.nn as nn
import torch.nn.functional as F
import torch

class Simu_Net(nn.Module):
    def __init__(self):
        super(Simu_Net,self).__init__()
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

class new_Simu_Net(Simu_Net):
    def __init__(self,model):
        super(new_Simu_Net,self).__init__()
        self.model=model
        self.fc3=nn.Linear(10,3)
        
    def forward(self,x):
        out = self.model(x)
        out = self.fc3(out) # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1) # 计算log(softmax(x))
        return out

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

def load_data():
    with open('MNIST_label.pkl', 'rb') as f:
        k_labels = pickle.load(f).astype(np.int64) 
    with open('MNIST_data.pkl', 'rb') as f2:
        k_data = pickle.load(f2).astype(np.float32)
    k_data = k_data/255.0
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
    
    return k_data, k_labels, test_dataset

batch_size = 300
epoch=5

l1=8
with open(f'MNIST_attack_model{l1}{l1+1}.pkl', 'rb') as f:
    load_model=pickle.load(f)
f.close()

model = new_Simu_Net(load_model).cpu()


l2=l1+1
k_data, k_labels, test_dataset= load_data()

for lab in range(len(k_labels)):
    if k_labels[lab]==l1:
        k_labels[lab]=0
    elif k_labels[lab]==l2:
        k_labels[lab]=1
    else:
        k_labels[lab]=2
        
for e in range(epoch):
    optimizer_b = torch.optim.Adam(model.parameters())
    model.train()
    
    for i in range(100):
        data,target = get_random_data_targets(k_data, k_labels, batch_size, 60000)
        data = data.reshape(300, 1, 28, 28)
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        optimizer_b.zero_grad()
        y_hat = model(data)
        loss = F.nll_loss(y_hat, target)
        loss.backward()
        optimizer_b.step()

        if(i+1)%30 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                e, i * len(data), 60000,
                100. * i / 60000, loss.item()))
            
    all_acc = []
    tem_set=deepcopy(test_dataset)
    for xx in range(len(tem_set)):
        if tem_set.targets[xx]==l1:
            tem_set.targets[xx]=0
        elif tem_set.targets[xx]==l2:
            tem_set.targets[xx]=1
        else:
            tem_set.targets[xx]=2
    test_loader = Data.DataLoader(dataset=tem_set, batch_size=batch_size, shuffle=True)
    
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        if (batch_idx>10):
            break
        data = data.reshape(300, 1, 28, 28)
        y_hat = model(data)
        y_t=(y_hat.argmax(axis=1))

        acc = y_t == target
        acc = acc + 0
        all_acc.append(acc.sum()/len(acc))
    print(f"Test acc: {(sum(all_acc) / len(all_acc)):.4f}") 

with open(f'MNIST_attack_model{l1}{l2}_improved.pkl', 'wb') as file:
    pickle.dump(model, file)



