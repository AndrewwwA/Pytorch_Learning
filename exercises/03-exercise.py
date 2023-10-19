import torch 
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

dive = 'cuda' if torch.cuda.is_available() else 'cpu'

# Importing MNIST dataset
train_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

# Visualize FIRST 5 different samples of the MNIST training dataset.
import matplotlib.pyplot as plt
class_names = train_data.classes
# print(class_names)
for i in range(0): # 0 for time purposes change to 5 to visualize
    img = train_data[i + 1][0]
    # print(img.shape)
    # torch.Size([1, 28, 28])
    img_squeeze = img.squeeze()
    # print(img_squeeze.shape)
    # torch.Size([28, 28])
    label = train_data[i + 1][1]
    plt.figure(figsize=(3, 3))
    plt.imshow(img_squeeze, cmap="gray")
    plt.title(label)
    plt.axis(False);
    

# Turn the MNIST train and test datasets into dataloaders 
# Turning into dataloaders
torch.manual_seed(42)
train_batches = DataLoader(
    batch_size=32,
    dataset=train_data,
    shuffle=True,
)
torch.manual_seed(42)
test_batches = DataLoader(
    batch_size=32,
    dataset=test_data,
    shuffle=True
)

class MNISTTinyVGG(nn.Module):
    def __init__(self,
                 hidden_channels):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) 
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      padding=1,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.SMBlock3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=490,
                      out_features=len(class_names)),
        )
    def forward(self, x):
        x = self.block1(x)
        print(x.shape)
        x = self.block2(x)
        print(x.shape)
        x = self.SMBlock3(x)
        print(x.shape)
        return x

Model_1 = MNISTTinyVGG(hidden_channels=10)
# print(train_data[0][0].shape)
# Model_1(train_data[0][0])