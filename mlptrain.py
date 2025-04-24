# MLP  
import os
import torch  
from torch import nn  
import torchvision  
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 下载FashionMNIST数据集  
fmnist_train = torchvision.datasets.FashionMNIST( 
   root='./data', train=True, download=True, transform=transforms.ToTensor())  

fmnist_text = torchvision.datasets.FashionMNIST(  
   root='./data', train=False, download=True, transform=transforms.ToTensor())  

#print(type(fmnist_train))  # 输出数据集类型)
#print(len(fmnist_train),len(fmnist_text))  
batch_size=256
epochs=20
lr = 0.1

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(
    fmnist_train, batch_size=batch_size, shuffle=True)
text_loader = torch.utils.data.DataLoader(
    fmnist_text, batch_size=batch_size, shuffle=False)

# 显示图像
#image, label = train_loader.__iter__().__next__()  
#print(image.shape, label.shape)

#plt.imshow(image[0].squeeze(), cmap="gray")
#plt.title(f"Label: {label[0].item()}")
#plt.show()

# 定义MLP模型
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)
# 定义损失函数和优化器
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
# 训练模型
def train(net, train_loader, optimizer, loss, epoch): #@save
    net.train()
    train_loss=0
    for i in range(epoch):
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss += l.item()*X.size(0)
        train_loss = train_loss/len(train_loader.dataset)
        print(f"Epoch {i+1}/{epoch}, Loss: {train_loss:.4f}")

if __name__ == "__main__":
    train(net, train_loader, optimizer, loss, epochs)
    save_path = "./FahionMLP.pkl"
    torch.save(net, save_path)




    



