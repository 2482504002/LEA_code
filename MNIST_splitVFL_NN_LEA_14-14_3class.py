from torchvision import datasets, transforms
import torch.utils.data as Data
import torch
import numpy as np
from copy import deepcopy
from MNIST_3kmeans import *
import random
import torch.nn.functional as F
import time
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
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

class bottle_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5) # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = nn.Conv2d(10, 20, 3) # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20*3*10, 500,bias=False) # 输入通道数是2000，输出通道数是500
        #self.fc2 = nn.Linear(500, 10) # 输入通道数是500，输出通道数是10，即10分类
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
        #out = self.fc2(out) # batch*500 -> batch*10
        #out = F.log_softmax(out, dim=1) 
        return out

class top_Net(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.fc2 = nn.Linear(n*500, 10,bias=False) # 输入通道数是500，输出通道数是10，即10分类
        self.fc2.weight.data=torch.clamp(self.fc2.weight.data, 0,1)
        #self.fc2.weight.data=torch.clamp(self.fc2.weight.data, 0, 1)
    def forward(self,x):
        out = self.fc2(x) # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1) # 计算log(softmax(x))
        return out

class Simu_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5) # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = nn.Conv2d(10, 20, 3) # 输入通道数10，输出通道数20，核的大小3
        
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20*3*10, 500,bias=False) # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(500, 10,bias=False) # 输入通道数是500，输出通道数是10，即10分类
        self.fc2.weight.data=torch.clamp(self.fc2.weight.data, 0,1)
        #self.fc2.weight.data=torch.clamp(self.fc2.weight.data, 0, 1)
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

def load_data():
    with open('MNIST_label.pkl', 'rb') as f:
        k_labels = pickle.load(f).astype(np.int64) 
    with open('MNIST_data.pkl', 'rb') as f2:
        k_data = pickle.load(f2).astype(np.float32)
    k_data = k_data/255.0
    
    k_data_c=k_data[:,:14,:]
    k_data_s=k_data[:,14:,:]
    
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
    test_dataset.data=test_dataset.data[:,:14,:]
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return k_data, k_labels, test_loader,k_data_c,k_data_s

def get_random_data_targets(k_data, k_labels, batch_size):
    k_data_list, k_labels_list=[], []
    selected_numbers = random.sample([x for x in range(len(k_data))], 300)
    for i in selected_numbers:
        k_data_list.append(k_data[i])
        k_labels_list.append(k_labels[i])
    data = np.stack(k_data_list)
    data = np.squeeze(data).reshape(batch_size, 392)
    target=np.array(k_labels_list)
    return data,target

def compare_matrices(matrix1, matrix2):
    #matrix1 (torch.Tensor): 形状为 [20, 5] 的第一个矩阵。
    #matrix2 (torch.Tensor): 形状为 [20, 5] 的第二个矩阵。
    frobenius_diff=0
    cos_sim=0
    for i in range(len(matrix1)):
        frobenius_diff += torch.norm(torch.from_numpy(matrix1[i]) - torch.from_numpy(matrix2[i]), p='fro')
        cos_sim += F.cosine_similarity(torch.from_numpy(matrix1[i]).view(-1), torch.from_numpy(matrix2[i]).view(-1), dim=0)
    cos_sim=cos_sim/len(matrix1)
    return (cos_sim, frobenius_diff.item())



start = time.time()
batch_size = 300
epoch=3

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_c = bottle_Net().to(DEVICE)
cc_layers=net_c.state_dict()
net_s = bottle_Net().to(DEVICE)
net_t=top_Net(2).to(DEVICE)
optimizer_b = torch.optim.Adam(net_c.parameters())
optimizer_s = torch.optim.Adam(net_s.parameters())
optimizer_t = torch.optim.Adam(net_t.parameters())



k_data, k_labels, test_loader,k_data_c,k_data_s= load_data()
m=len(k_data_c)
for e in range(epoch):
    net_c.train()
    net_s.train()
    net_t.train()
    for i in range(m//batch_size):
        data,target = get_random_data_targets(k_data_c, k_labels, batch_size)
        data1,target1 = get_random_data_targets(k_data_s, k_labels, batch_size)
        
        data = data.reshape(300, 1, 14, 28)
        data1 = data1.reshape(300, 1, 14, 28)
        data, target = torch.from_numpy(data).to(DEVICE), torch.from_numpy(target).to(DEVICE)
        data1, target1 = torch.from_numpy(data1).to(DEVICE), torch.from_numpy(target1).to(DEVICE)
        optimizer_b.zero_grad()
        optimizer_t.zero_grad()
        optimizer_s.zero_grad()
        output1 = net_c(data)
        output2 = net_s(data1)
        y_hat=torch.cat((output1,output2),axis=1)
        y_hat=net_t(y_hat)
        #y_hat = net_t(output)
        #y_hat=F.log_softmax(output1+output2, dim=1)
        loss = F.nll_loss(y_hat, target)
        loss.backward()
        if e== 0:
             detac=deepcopy(net_c.fc1.weight.grad.data)
        
        #net_c.fc1.bias.grad*=net_t[0]
        optimizer_b.step()
        optimizer_s.step()
        optimizer_t.step()
        if(i+1)%30 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                e, i * len(data), m,
                100. * i / m, loss.item()))

'''    test_loss = 0
    correct = 0
    all_acc = []
    net_c.eval()
    net_s.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output1 = net_c(data)
            output2 = net_s(data1)
            y_hat=F.log_softmax(output1+output2, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
 
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))'''


c_layers=deepcopy(cc_layers)
c_layers["conv1.weight"]=c_layers["conv1.weight"].reshape(10, 25).cpu()
cv1=np.hstack((c_layers["conv1.weight"], c_layers["conv1.bias"].reshape(10, 1).cpu()))
c_layers["conv2.weight"]=c_layers["conv2.weight"].reshape(20, 90).cpu()
cv2=np.hstack((c_layers["conv2.weight"], c_layers["conv2.bias"].reshape(20, 1).cpu()))
#cv3=np.hstack((c_layers["fc1.weight"].cpu(), c_layers["fc1.bias"].reshape(500, 1).cpu()))
cv3=np.array(c_layers["fc1.weight"].cpu())
#cv4=np.hstack((c_layers["fc2.weight"].cpu(), c_layers["fc2.bias"].reshape(10, 1).cpu()))
'''
cc_layers["fc2.weight"]=c_layers["fc2.weight"]
cc_layers["fc2.bias"]=c_layers["fc2.bias"]'''
bottle_layer_c=[cv1,cv2,cv3]
#bottle_layer_c=[cv3]


batch_size = 300
epoch=3
class_num=3
l1=0
l2=l1+1
l3=l2+1

k_data, k_labels, test_loader,k_data_c,k_data_s = load_data()
kk_data, kk_labels=[], []
b1=np.where(k_labels==l1)
kk_data1=np.array([k_data_c[idx] for idx in b1[0]])
kk_labels1=np.array([k_labels[idx] for idx in b1[0]])

b2=np.where(k_labels==l2)
kk_data2=np.array([k_data_c[idx] for idx in b2[0]])
kk_labels2=np.array([k_labels[idx] for idx in b2[0]])

b3=np.where(k_labels==l3)
kk_data3=np.array([k_data_c[idx] for idx in b3[0]])
kk_labels3=np.array([k_labels[idx] for idx in b3[0]])

kk_data=np.vstack((kk_data1, kk_data2, kk_data3))
kk_labels=np.hstack((kk_labels1, kk_labels2, kk_labels3))
len1=len(b1[0])
len2=len(b2[0])
len3=len(b3[0])

simulate_labels_list=[]
nets_list=[]
for i in range(l1,l1+class_num):
    for j in range(l1,l1+class_num):
        if i==j:
            continue
        simulate_labels = np.hstack((np.full(len1, i), np.full(len2, j), np.full(len3, 3-i-j)))
        simulate_labels_list.append(simulate_labels)
        simulate_net=Simu_Net().to(DEVICE)
        #simulate_net.load_state_dict(cc_layers)
        simulate_net.conv1.weight.data=deepcopy(cc_layers['conv1.weight'])
        simulate_net.conv1.bias.data=deepcopy(cc_layers['conv1.bias'])
        simulate_net.conv2.weight.data=deepcopy(cc_layers['conv2.weight'])
        simulate_net.conv2.bias.data=deepcopy(cc_layers['conv2.bias'])
        simulate_net.fc1.weight.data=deepcopy(cc_layers['fc1.weight'])
        #simulate_net.fc1.bias.data=deepcopy(cc_layers['fc1.bias'])
        if len(nets_list)!=0:
            simulate_net.fc2.weight.data=deepcopy(nets_list[0].fc2.weight.data)
            #simulate_net.fc2.bias.data=deepcopy(nets_list[0].fc2.bias.data)
        
        #simulate_net.layers=net_c.layers+net_s.layers
        nets_list.append(simulate_net)
net_num=len(nets_list)
label_num=len1+len2+len3

bottle_layers=[]
for p in range(net_num):
    tem_labels=simulate_labels_list[p]
    simulate_net=deepcopy(nets_list[p])
    optimizer_simu = torch.optim.Adam(simulate_net.parameters())
    for e in range(epoch):
        for i in range(label_num//batch_size):
            data,target = get_random_data_targets(kk_data, tem_labels, batch_size)
            data = data.reshape(300, 1, 14, 28)
            data, target = torch.from_numpy(data).to(DEVICE), torch.from_numpy(target).type(torch.int64).to(DEVICE)
            optimizer_simu.zero_grad()
            output = simulate_net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            if p == 0:
             deta1=deepcopy(simulate_net.fc1.weight.grad.data)
            if p == 1:
             deta2=deepcopy(simulate_net.fc1.weight.grad.data)
            if p == 2:
             deta3=deepcopy(simulate_net.fc1.weight.grad.data)
            if p == 3:
             deta4=deepcopy(simulate_net.fc1.weight.grad.data)
            if p == 4:
             deta5=deepcopy(simulate_net.fc1.weight.grad.data)
            if p == 5:
             deta6=deepcopy(simulate_net.fc1.weight.grad.data)
            optimizer_simu.step()
            if(i+1)%30 == 0: 
                print('Simu:{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(p,
                    e, i * len(data), label_num,
                    100. * i / label_num, loss.item()))
   
    nets_list[p]=simulate_net
    
    si_layers=simulate_net.state_dict()
    c_layers=deepcopy(si_layers)
    c_layers["conv1.weight"]=c_layers["conv1.weight"].reshape(10, 25).cpu()
    cv1=np.hstack((c_layers["conv1.weight"], c_layers["conv1.bias"].reshape(10, 1).cpu()))
    c_layers["conv2.weight"]=c_layers["conv2.weight"].reshape(20, 90).cpu()
    cv2=np.hstack((c_layers["conv2.weight"], c_layers["conv2.bias"].reshape(20, 1).cpu()))
    #cv3=np.hstack((c_layers["fc1.weight"].cpu(), c_layers["fc1.bias"].reshape(500, 1).cpu()))
    cv3=np.array(c_layers["fc1.weight"].cpu())
    #cv4=np.hstack((c_layers["fc2.weight"].cpu(), c_layers["fc2.bias"].reshape(10, 1).cpu()))
    bottle_layer_s=[cv1,cv2,cv3]
    #bottle_layer_s=[cv3]
    bottle_layers.append(bottle_layer_s)
c_l=[]
for x in range(net_num):
    c=compare_matrices(bottle_layer_c,bottle_layers[x])
    c_l.append([c,x])
print(c_l)

print(compare_matrices(bottle_layers[1],bottle_layers[0]))

attack_idx=c_l.index(max(c_l))
c,x1=c_l[attack_idx][0],c_l[attack_idx][1]
print(f"Get attack model:{simulate_labels_list[x1][0]}{simulate_labels_list[x1][-1]}")
attack_net=deepcopy(nets_list[x1])

from torch.nn.functional import cosine_similarity
c1=cosine_similarity(detac, deta1,dim=0).sum()
c2=cosine_similarity(detac, deta2,dim=0).sum()
c3=cosine_similarity(detac, deta3,dim=0).sum()
c4=cosine_similarity(detac, deta4,dim=0).sum()
c5=cosine_similarity(detac, deta5,dim=0).sum()
c6=cosine_similarity(detac, deta6,dim=0).sum()
   
print("c1: ",c1,l1)
print("c2: ",c2,l2)
print("c3: ",c3,l2)
print("c4: ",c4,l2)
print("c5: ",c5,l2)
print("c6: ",c6,l2)


test_loss = 0
correct = 0
attack_net.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = attack_net(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
        pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

'''with open(f'MNIST_attack_model{simulate_labels_list[x1][0]}{simulate_labels_list[x1][-1]}.pkl', 'wb') as file:
    pickle.dump(attack_net, file)'''

end = time.time()
print('总用时：'+str(end-start)+'秒！')

