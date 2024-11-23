from torchvision import datasets, transforms
import torch.utils.data as Data
import torch
import torchvision
from torch.utils.data import Subset
import numpy as np
from copy import deepcopy
from MNIST_3kmeans import *
import random
import torch.nn.functional as F
import time
import torch.nn as nn
import torch.nn.functional as F
from MyNet import CustomResNet18,bottle_ResNet18,top_ResNet18,simu_ResNet18
import itertools
import torch.optim as optim

def eval(custom1_resnet18, custom2_resnet18,labels_of_interest):
    custom1_resnet18.eval()
    custom2_resnet18.eval()

    # 数据处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    indices = [i for i, label in enumerate(testset.targets) if label in labels_of_interest]
    testset = Subset(testset, indices)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            
            inputs1=images[:,:,:18,:]
            inputs2=images[:,:,18:,:]
            
            outputs1 = custom1_resnet18(inputs1)
            outputs2 = custom2_resnet18(inputs2)
            _, predicted = torch.max((outputs1+outputs2).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('模型准确率: %d %%' % (100 * correct / total))
    
def simu_eval(model,dataset):
    model.eval()
    # 数据处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    indices = [i for i, label in enumerate(testset.targets) if label in labels_of_interest]
    testset = Subset(testset, indices)
    
    testloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            
            inputs1=images[:,:,:18,:]
            
            outputs1 = model(inputs1)
            _, predicted = torch.max((outputs1).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('训练准确率: %d %%' % (100 * correct / total))
    return 100 * correct / total

def simu_test_eval(model,labels_of_interest):
    model.eval()
    # 数据处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    indices = [i for i, label in enumerate(testset.targets) if label in labels_of_interest]
    testset = Subset(testset, indices)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            
            inputs1=images[:,:,:18,:]
            
            outputs1 = model(inputs1)
            _, predicted = torch.max((outputs1).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('训练准确率: %d %%' % (100 * correct / total))
    return 100 * correct / total

classes=10
labels_of_interest = [i for i in range(classes)]

#k_data, k_labels, test_loader = load_data()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
indices = [i for i, label in enumerate(trainset.targets) if label in labels_of_interest]
trainset5 = Subset(trainset, indices)
trainloader1 = torch.utils.data.DataLoader(trainset5, batch_size=64, shuffle=False)

cluster_acc=0.93
err_num=np.random.randint(0,len(trainset5.dataset.targets),size=int(len(trainset5.dataset.targets)*(1-cluster_acc)))
trainset5_cluster=deepcopy(trainset5)
for j in err_num:
    trainset5_cluster.dataset.targets[j]=(trainset5_cluster.dataset.targets[j]+1)%10
#trainloader = torch.utils.data.DataLoader(trainset5_cluster, batch_size=256, shuffle=False)

classes1=deepcopy(classes)


custom1_resnet18 = CustomResNet18()

custom2_resnet18 = CustomResNet18()
custom1_resnet18.to(device)
custom2_resnet18.to(device)
optimizer1 = optim.SGD(custom1_resnet18.parameters(), lr=0.001, momentum=0.9)
optimizer2 = optim.SGD(custom2_resnet18.parameters(), lr=0.001, momentum=0.9)
custom1_resnet18_copy=deepcopy(custom1_resnet18)
running_loss = 0.0

for epoch in range(1):
    custom1_0resnet18=deepcopy(custom1_resnet18)
    for i, data in enumerate(trainloader1, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        inputs1=inputs[:,:,:18,:]
        inputs2=inputs[:,:,18:,:]
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        outputs1 = custom1_resnet18(inputs1)
        outputs2 = custom2_resnet18(inputs2)
        
        loss = criterion(outputs1+outputs2, labels)
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        running_loss += loss.item()
        if i % 200 == 199:    # 每200个batch打印一次
            print('[%5d] loss: %.3f' %
                    (i + 1, running_loss / 2000))
            running_loss = 0.0
    if epoch==0:
        params_list1=list(custom1_resnet18.parameters())[:-2]
        params_numpy1 = [param.cpu().detach().numpy() for param in params_list1]
        grad_1 = np.concatenate([param.ravel() for param in params_numpy1])
        params_list1=list(custom1_0resnet18.parameters())[:-2]
        params_numpy1 = [param.cpu().detach().numpy() for param in params_list1]
        grad_2 = np.concatenate([param.ravel() for param in params_numpy1])
        grad_1=torch.from_numpy(grad_1).to(device)-torch.from_numpy(grad_2).to(device)
        #grad_cv1_w=custom1_resnet18.resnet18.conv1.weight.data-custom1_0resnet18.resnet18.conv1.weight.data
        #grad_1=grad_1.reshape(64*7*7)
        #grad_1=deepcopy(custom1_resnet18.resnet18.conv1.weight.grad.data)
print('Finished Training')
eval(custom1_resnet18, custom2_resnet18,labels_of_interest)

start=time.time()
attack_models=[]
for ii in range(classes//2):#迭代一轮生成一个三输出的攻击模型
    all_possible=[]
    simu_models=[]
    simu_datasets=[]
    all_grad=[]
    all_possible_arrays = list(itertools.product(range(0+ii*2,classes), repeat=2))
    for i in range(len(all_possible_arrays)):
        if all_possible_arrays[i][0]!=all_possible_arrays[i][1]:
            all_possible.append(all_possible_arrays[i])
        simu_model=deepcopy(custom1_resnet18_copy)
        simu_models.append(simu_model)
    
    label0=0
    label1=1
    label2=2

    
    for possible_array in all_possible:
        simu_dataset=deepcopy(trainset5_cluster)
        for j in simu_dataset.indices:
            if simu_dataset.dataset.targets[j] == possible_array[0]:
                simu_dataset.dataset.targets[j]=label0
            elif simu_dataset.dataset.targets[j] == possible_array[1]:
                simu_dataset.dataset.targets[j]=label1
            elif simu_dataset.dataset.targets[j]!=label0 and simu_dataset.dataset.targets[j]!=label1:
                simu_dataset.dataset.targets[j]=label2
        simu_datasets.append(simu_dataset)
    
    for i in range(len(all_possible)):
        simu_model=simu_models[i]
        simu_model.resnet18.fc = nn.Linear(simu_model.resnet18.fc.in_features, 3)
        simu_model0=deepcopy(simu_model)
        simu_model.to(device)
        simu_optimizer = optim.SGD(simu_model.parameters(), lr=0.001, momentum=0.9)
        running_loss = 0.0
        simu_dataset=simu_datasets[i]
        trainloader = torch.utils.data.DataLoader(simu_dataset, batch_size=64, shuffle=False)
        for epoch in range(1):
            for iii, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs1=inputs[:,:,:18,:]
                simu_optimizer.zero_grad()
                outputs1 = simu_model(inputs1)
                loss = criterion(outputs1, labels)
                loss.backward()
                simu_optimizer.step()
                running_loss += loss.item()
                if iii % 200 == 199:    # 每200个batch打印一次
                    print('[%5d] loss: %.3f' %
                            (iii + 1, running_loss / 2000))
                    running_loss = 0.0
            if epoch==0:
                params_list=list(simu_model.parameters())[:-2]
                params_numpy = [param.cpu().detach().numpy() for param in params_list]
                grad1 = np.concatenate([param.ravel() for param in params_numpy])
                params_list=list(simu_model0.parameters())[:-2]
                params_numpy = [param.cpu().detach().numpy() for param in params_list]
                grad2 = np.concatenate([param.ravel() for param in params_numpy])
                #grad=simu_model.resnet18.conv1.weight.data-simu_model0.resnet18.conv1.weight.data
                #grad=grad.reshape(64*7*7)
                grad=torch.from_numpy(grad1).to(device)-torch.from_numpy(grad2).to(device)
                all_grad.append(grad)
        #acc=simu_eval(simu_model,simu_dataset)
        print(f"Simu_model {i} finished.")
        
        simu_eval(simu_model,simu_dataset)
        simu_test_eval(simu_model,labels_of_interest)
        
        
    score=[0]*len(all_possible)
    for i in range(len(all_possible)):
        score[i]=F.cosine_similarity(grad_1, all_grad[i],dim=0).sum()
    max_score=max(score)
    max_index=score.index(max_score)
    attack_model=simu_models[max_index]
    attack_models.append(simu_models[0])
    with open(f'./Binary_resnet18_{label0}{label1}.pkl', 'wb') as file:
        pickle.dump(simu_models[0], file)

    print(score)


end=time.time()
print(f"10class Time cost: {end-start} s")
# 加载模型
#for i in range(classes//2):
    
    #with open(f'Binary_resnet18_{i*2}{i*2+1}.pkl', 'rb') as f:
     #   attack_model=pickle.load(f)
       # attack_model.eval()
       # attack_models.append(attack_model)
# 数据处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])
labels_of_interest = [i for i in range(classes)]

testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
indices = [i for i, label in enumerate(testset.targets) if label in labels_of_interest]
testset5 = Subset(testset, indices)
testloader1 = torch.utils.data.DataLoader(testset5, batch_size=64, shuffle=False)


correct = 0
total = 0
with torch.no_grad():
    for data in testloader1:
        images, labels = data[0].to(device), data[1].to(device)
        inputs1=images[:,:,:18,:]
        outputs=[]
        for i in range(classes//2):
            outputs1=attack_models[i](inputs1)
            
            _, predicted = torch.max((outputs1).data, 1)
            outputs.append(predicted)
        pre_label=[]
        for x in range(labels.size(0)):
            if outputs[0][x]==0:
                pre_label.append(0)
            if outputs[0][x]==1:
                pre_label.append(1)
            if outputs[0][x]==2:
                if outputs[1][x]==0:
                    pre_label.append(2)
                if outputs[1][x]==1:
                    pre_label.append(3)
                if outputs[1][x]==2:
                    if outputs[2][x]==0:
                        pre_label.append(4)
                    if outputs[2][x]==1:
                        pre_label.append(5)
                    if outputs[2][x]==2:
                        if outputs[3][x]==0:
                            pre_label.append(6)
                        if outputs[3][x]==1:
                            pre_label.append(7)
                        if outputs[3][x]==2:
                            if outputs[4][x]==0:
                                pre_label.append(8)
                            if outputs[4][x]==1:
                                pre_label.append(9)
                            if outputs[4][x]==2:
                                pre_label.append(9)
        total += labels.size(0)
        correct += (torch.tensor(pre_label).to(device) == labels).sum().item()

print('模型准确率: %d %%' % (100 * correct / total))

