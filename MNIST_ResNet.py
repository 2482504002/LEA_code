'''import torch
from torchvision.models import resnet18
import torch.nn as nn
import torchvision.models as models


# 创建自定义的ResNet-18模型类
class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入通道数为1
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 10)  # 修改输出大小为10（适应10个类别）

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet18.fc(x)

        return x
    
# # 创建自定义ResNet-18模型实例
# custom_resnet18 = CustomResNet18()

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# 数据处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)


# 创建自定义ResNet-18模型实例
custom_resnet18 = CustomResNet18()

# 使用GPU进行训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
custom_resnet18.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(custom_resnet18.parameters(), lr=0.001, momentum=0.9)
#Adam优化
# optimizer = torch.optim.Adam(custom_resnet18.parameters(), lr=0.001)

# 训练模型
for epoch in range(5):  # 遍历数据集多次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        
        outputs = custom_resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:    # 每200个batch打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# 保存训练好的模型
PATH = './mnist_resnet18.pth'
torch.save(custom_resnet18.state_dict(), PATH)
'''
import torch
import torchvision.transforms as transforms
import torchvision
from MyNet import CustomResNet18

# 使用GPU进行训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 创建自定义ResNet-18模型实例
custom_resnet18 = CustomResNet18().to(device)
custom_resnet18.load_state_dict(torch.load('mnist_resnet18.pth'))

def eval(custom_resnet18):
    custom_resnet18.eval()

    # 数据处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = custom_resnet18(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('模型准确率: %d %%' % (100 * correct / total))
    
eval(custom_resnet18)
