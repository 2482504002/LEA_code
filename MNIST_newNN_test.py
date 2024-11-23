from torchvision import datasets, transforms
import torch.utils.data as Data
import numpy as np
from copy import deepcopy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(threshold=np.inf)

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
epoch=3
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)


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
    
test_loss = 0
correct = 0
model01.eval()
model23.eval()
model45.eval()
model67.eval()
model89.eval()
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
acc=0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output1 = model01(data)
        output2 = model23(data)
        output3 = model45(data)
        output4 = model67(data)
        output5 = model89(data)
        all_hat=[output1,output2,output3,output4,output5]
        simu_label=[]
        for idx in range(100):
            max_id=[]
            for hat in all_hat:
                max_id.append([max(hat[idx]), np.argmax(hat[idx].cpu())])
            simu_label.append(max_id[max_id.index(max(max_id))][1])
        simu_label=np.array(simu_label)
        
        for x in range(len(simu_label)):
            if simu_label[x]==target[x]:
                acc+=1
accc=acc/10000
#acc = (simu_label == np.array(target.cpu()))
#acc = acc + 0
#accc=acc.mean()
print(accc)
a=1