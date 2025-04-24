#CNN
import torch
import os
from torch import nn
import torchvision  
from torchvision import transforms
import matplotlib.pyplot as plt
from mlptrain import train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fmnist_train = torchvision.datasets.FashionMNIST( 
   root='./data', train=True, download=True, transform=transforms.ToTensor())  

fmnist_text = torchvision.datasets.FashionMNIST(  
   root='./data', train=False, download=True, transform=transforms.ToTensor())

batch_size = 256
epochs = 20
lr=1e-4

train_loader = torch.utils.data.DataLoader(
    fmnist_train, batch_size=batch_size, shuffle=True)
text_loader = torch.utils.data.DataLoader(
    fmnist_text, batch_size=batch_size, shuffle=False)
# 定义CNN模型
net_ = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 卷积层
    nn.ReLU(),  # 激活函数
    nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 卷积层
    nn.ReLU(),  # 激活函数
    nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层
    nn.Flatten(),  # 展平层
    nn.Linear(64 * 7 * 7, 128),  # 全连接层
    nn.ReLU(),  # 激活函数
    nn.Linear(128, 10)  # 输出层
).to(device)
# 定义损失函数和优化器
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_.parameters(), lr)
# 训练模型
train(net_, train_loader, optimizer, loss, epochs)
save_path = "./FahionCNN.pkl"
torch.save(net_, save_path)
