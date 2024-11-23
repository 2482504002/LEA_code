import torch
import torch.nn as nn
from MyNet import CustomResNet18,bottle_ResNet18,simu_ResNet18

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Subset
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
import itertools

def simu_eval(model,labels_of_interest):
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

    print('模型准确率: %d %%' % (100 * correct / total))
    return 100 * correct / total

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
            
            inputs1=images[:,:,:20,:]
            inputs2=images[:,:,20:,:]
            
            outputs1 = custom1_resnet18(inputs1)
            outputs2 = custom2_resnet18(inputs2)
            _, predicted = torch.max((outputs1+outputs2).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('模型准确率: %d %%' % (100 * correct / total))
    

classes=3
labels_of_interest = [i for i in range(classes)]


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
indices = [i for i, label in enumerate(trainset.targets) if label in labels_of_interest]
trainset = Subset(trainset, indices)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False)


permutations = list(itertools.permutations(labels_of_interest))
simu_num=len(permutations)

custom1_resnet18 = simu_ResNet18()
simu_bottle=deepcopy(custom1_resnet18)
simu_models=[]
grad=[0]*simu_num
custom2_resnet18 = simu_ResNet18()

custom1_resnet18.to(device)
custom2_resnet18.to(device)
optimizer1 = optim.SGD(custom1_resnet18.parameters(), lr=0.001, momentum=0.9)
optimizer2 = optim.SGD(custom2_resnet18.parameters(), lr=0.001, momentum=0.9)


running_loss = 0.0
for epoch in range(1):
    custom1_0resnet18=deepcopy(custom1_resnet18)
    for i, data in enumerate(trainloader, 0):
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
        grad_1=custom1_resnet18.resnet18.conv1.weight.data-custom1_0resnet18.resnet18.conv1.weight.data
        grad_1=grad_1.reshape(64*7*7)
        #grad_1=deepcopy(custom1_resnet18.resnet18.conv1.weight.grad.data)
print('Finished Training')
PATH1 = './mnist_1resnet18.pth'
PATH2 = './mnist_2resnet18.pth'
torch.save(custom1_resnet18.state_dict(), PATH1)
torch.save(custom2_resnet18.state_dict(), PATH2)




eval(custom1_resnet18,custom2_resnet18,labels_of_interest)


import   time
cluster_acc=0.90
err_num=np.random.randint(0,len(trainset.dataset.targets),size=int(len(trainset.dataset.targets)*(1-cluster_acc)))
for j in err_num:
    trainset.dataset.targets[j]=(trainset.dataset.targets[j]+1)%10
trainset1=deepcopy(trainset)
start=time.time()
for epoch in range(1): 
    
    for n in range(simu_num):
        simu_model=deepcopy(simu_bottle)
        
        simu_model.to(device)
        optimizer= optim.SGD(simu_model.parameters(), lr=0.001, momentum=0.9)
        running_loss = 0.0

        #生成模拟标签
        simulabel_map=permutations[n]
        value_map = {i:simulabel_map[i] for i in range(classes)}
        #value_map = {0: simulabel_map[0], 1: simulabel_map[1], 2: simulabel_map[2]}#, 3: simulabel_map[3], 4: simulabel_map[4]
        list_target=trainset.dataset.targets.tolist()
        #labels=
        new_list = torch.tensor([value_map.get(item, item) for item in list_target])
        trainset1.dataset.targets =  new_list
        trainloader = torch.utils.data.DataLoader(trainset1, batch_size=256, shuffle=False)
        simu_0model=deepcopy(simu_model)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs1=inputs[:,:,:20,:]
            optimizer.zero_grad()
            outputs1 = simu_model(inputs1)
            loss = criterion(outputs1, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:    # 每200个batch打印一次
                print('[%5d] loss: %.3f' %
                        (i + 1, running_loss / 2000))
                running_loss = 0.0
        print(f'Simu{n} Finished Training')
        acc=simu_eval(simu_model,labels_of_interest)
        simu_models.append(simu_model)
        if epoch==0:
            grad[n]=simu_model.resnet18.conv1.weight.data-simu_0model.resnet18.conv1.weight.data
            grad[n]=grad[n].reshape(64*7*7)
            #grad[n]=deepcopy(simu_model.resnet18.conv1.weight.grad.data)
    if epoch==0:
        score=[0]*simu_num
        for ii in range(simu_num):
            score[ii]=F.cosine_similarity(grad_1, grad[ii],dim=0).sum()
        max_score=max(score)
        max_index=score.index(max_score)
        attack_model=simu_models[max_index]
        print(score)
        acc=simu_eval(attack_model,labels_of_interest)
        print("attack acc:", acc)
        
end=time.time()
print(f"5class Time cost: {end-start} s")
# 创建自定义ResNet-18模型实例

