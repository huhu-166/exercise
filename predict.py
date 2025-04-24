import os
import torch  
from torch import nn  
import torchvision  
from torchvision import transforms
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fmnist_text = torchvision.datasets.FashionMNIST(  
   root='./data', train=False, download=True, transform=transforms.ToTensor()) 
text_loader = torch.utils.data.DataLoader(
    fmnist_text, batch_size=256, shuffle=False)


# 加载模型
loaded_net = torch.load("./FahionCNN.pkl", weights_only=False)
loaded_net.eval()

# 定义预测函数
def predict(model, data_loader, device):
   model.to(device)
   predictions = []
   with torch.no_grad():
       for X, _ in data_loader:
           X = X.to(device)
           y_hat = model(X)
           predicted_labels = torch.argmax(y_hat, axis=1)
           predictions.extend(predicted_labels.cpu().numpy())
   return predictions

predictions = predict(loaded_net, text_loader, device)

# 显示前10张图片及其预测结果
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 获取前10张图片和预测结果
images, labels = next(iter(text_loader))
images = images[10:20]
predicted_labels = predictions[10:20]

# 绘制图片
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.title(f"Pred: {classes[predicted_labels[i]]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
