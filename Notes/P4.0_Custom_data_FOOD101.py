# %%
import torch
from torch import nn

devie = 'cuda' if torch.cuda.is_available() else 'cpu'

# Building a food vision using a subset of FOOD101 dataset \
    
from pathlib import Path
data_path = Path('../data')
image_path = data_path / "pizza_steak_shushi"

# EXPLORING DATASET
print(data_path)
import os
def walk_dir(dir_path):
    """Returns contents of dir_peth"""
    print('test')
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"there are {len(dirnames)} directories and {len(filenames)} images in {dirpath} "),
# walk_dir(image_path)
# there are 1 directories and 1 images in data\pizza_steak_shushi 
# there are 2 directories and 0 images in data\pizza_steak_shushi\pizza_steak_sushi
# there are 3 directories and 0 images in data\pizza_steak_shushi\pizza_steak_sushi\test
# there are 0 directories and 25 images in data\pizza_steak_shushi\pizza_steak_sushi\test\pizza
# there are 0 directories and 19 images in data\pizza_steak_shushi\pizza_steak_sushi\test\steak
# there are 0 directories and 31 images in data\pizza_steak_shushi\pizza_steak_sushi\test\sushi
# there are 3 directories and 0 images in data\pizza_steak_shushi\pizza_steak_sushi\train
# there are 0 directories and 78 images in data\pizza_steak_shushi\pizza_steak_sushi\train\pizza
# there are 0 directories and 75 images in data\pizza_steak_shushi\pizza_steak_sushi\train\steak
# there are 0 directories and 72 images in data\pizza_steak_shushi\pizza_steak_sushi\train\sushi

### Visualizing a RANDOM image of each type ###
from random import randrange
from PIL import Image #(PiLLOW)
import random


# random.seed(42)
# data\pizza_steak_shushi\pizza_steak_sushi

# Get all image paths
# pathhh = Path("../data/pizza_steak_shushi")
image_path_list = list(image_path.glob("*/*/*.jpg")) # Goes three layers down (each star) jpg
# print("The length of image_path_list is:", len(image_path_list))


# print(image_path_list) # Prints every single image path (NOt showing because space) (SHOWN AS A ARRAY)

# Pick range image path
random_path = randrange(len(image_path_list)) 
# print("The value of random_path is:", random_path)
# print(random_path) # data\pizza_steak_shushi\pizza_steak_sushi\test\sushi\2394442.jpg always since random.seed()

# Access class name from path name (said in directory)
image = image_path_list[random_path]
image_class = image_path_list[random_path].parent.stem
# print(image_class) # sushi


# Open image using PIL (PYTHON IMAGE LIBRARY)
img = Image.open(image)

#Can show img.width, img.height, imge.class
## SHOW USING MATPLOTLIB

import numpy as np
import matplotlib.pyplot as plt

# Turn the image into an array
img_as_array = np.asarray(img)

# Plot the image with matplotlib
# plt.figure(figsize=(9, 6))
# plt.imshow(img_as_array)
# plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
# plt.axis(False);


### Turning data into pytorch tensors ###
# Currently can't be used to train a model, using 'torch.utils.data.Dataset' then 'torch.utils.data.Dataloader'
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Using torchvision.transforms to change JPG To tensors
data_transform = transforms.Compose([
    # Resize images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the image randomly
    transforms.RandomHorizontalFlip(p=0.5),
    # Turn into tensor
    transforms.ToTensor()
])
# print(data_transform(img))

def plot_transed_images(image_path, transform, n=0, seed=None):
    if seed:
        random.seed(seed)

    random_image_paths = random.sample(image_path, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title("Pre Transform")
            transformed_image = transform(f).permute(1, 2, 0) # Change color channel to the end from (C, H, W) TO (H, W, C)
            ax[1].imshow(transformed_image)
            ax[1].set_title("transformed Image")
            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=14)

plot_transed_images(image_path_list, data_transform, n=3, seed=42)

### Loading image data using 'torchvision.datasets.ImageFolder' - creates datasets
from torchvision import datasets
# print(Path('data/pizza_steak_sushi').)
train_dir, test_dir = image_path / 'train', image_path / 'test'
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform, #What to do to the data (FUNCTION I CREATED),
                                  target_transform=None
                                  )
test_data = datasets.ImageFolder(root=test_dir,
                                transform=data_transform,
                                target_transform=None)
# print(train_data)
# Dataset ImageFolder
#     Number of datapoints: 225
#     Root location: ..\data\pizza_steak_shushi\train
#     StandardTransform
# Transform: Compose(
#                Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=warn)
#                RandomHorizontalFlip(p=0.5)
#                ToTensor()
#            )
from torch.utils.data import DataLoader

# Turn into dataloader/batch
train_batches = DataLoader(batch_size=1,
                         shuffle=True,
                         dataset=train_data,
)
test_batches = DataLoader(batch_size=1,
                        shuffle=False,
                        dataset=test_data)

### Creating TinyVGG model without data augmentation
simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])
### NON AUGMENT IMAGEFOLDER
simple_train_data = datasets.ImageFolder(
    root=train_dir,
    transform=simple_transform,
    target_transform=None
)
simple_test_data = datasets.ImageFolder(
    root=test_dir,
    transform=simple_transform,
    target_transform=None
)
### NON AUGMENT Batch
simple_train_batches = DataLoader(batch_size=32,
                                  shuffle=True,
                                  dataset=simple_train_data)
simple_test_batches = DataLoader(batch_size=32,
                                 dataset=simple_test_data,
                                 shuffle=False)

### TinyVGG model
class TinyVGG(nn.Module):
    def __init__(self, input, hidden_units, output):
        super().__init__()
        self.convLayer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input,
                out_channels=hidden_units,
                kernel_size=3,
                padding=0,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=0,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
            )
        self.convLayer2 = nn.Sequential(
                        nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=0,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=0,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
            )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=1690,
                out_features=3
            )
        )
        
    def forward(self, x):
        # x = self.convLayer1(x)
        # print(x.shape)
        # x = self.convLayer2(x)
        # print(x.shape)
        # x = self.classifier(x)
        # print(x)
        # return x
        return self.classifier(self.convLayer2(self.convLayer1(x)))

torch.manual_seed(42)
model_0 = TinyVGG(input=3, # color channels
                  hidden_units=10,
                  output=3 # Num of options in classifier
                  ).to('cuda')
# print(model_0)

### TEST
# dummy_data = torch.randn(3, 64, 64).to('cuda')
# dummy_data, label = next(iter(simple_train_batches))
# print(model_0(dummy_data.to('cuda')))

# %%
### Torchinfo to view params and other important information
import torchinfo 
from torchinfo import summary

# view specs of model
# summary(model_0, input_size=[1 #Batch of a singular image
#                              , 3, 64, 64])

# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# TinyVGG                                  [1, 3]                    --
# ├─Sequential: 1-1                        [1, 10, 30, 30]           --
# │    └─Conv2d: 2-1                       [1, 10, 62, 62]           280
# │    └─ReLU: 2-2                         [1, 10, 62, 62]           --
# │    └─Conv2d: 2-3                       [1, 10, 60, 60]           910
# │    └─ReLU: 2-4                         [1, 10, 60, 60]           --
# │    └─MaxPool2d: 2-5                    [1, 10, 30, 30]           --
# ├─Sequential: 1-2                        [1, 10, 13, 13]           --
# │    └─Conv2d: 2-6                       [1, 10, 28, 28]           910
# │    └─ReLU: 2-7                         [1, 10, 28, 28]           --
# │    └─Conv2d: 2-8                       [1, 10, 26, 26]           910
# │    └─ReLU: 2-9                         [1, 10, 26, 26]           --
# │    └─MaxPool2d: 2-10                   [1, 10, 13, 13]           --
# ├─Sequential: 1-3                        [1, 3]                    --
# │    └─Flatten: 2-11                     [1, 1690]                 --
# │    └─Linear: 2-12                      [1, 3]                    5,073
# ==========================================================================================
# Total params: 8,083
# Trainable params: 8,083
# Non-trainable params: 0
# Total mult-adds (M): 5.69
# ==========================================================================================
# Input size (MB): 0.05
# Forward/backward pass size (MB): 0.71
# Params size (MB): 0.03
# Estimated Total Size (MB): 0.79
# ==========================================================================================

model_0 = TinyVGG(input=3,
                  hidden_units=10,
                  output=len(train_data.classes)).to('cuda')

### Train and test step ###
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lr=0.001, params=model_0.parameters())

def train_step(model,
               dataloader: torch.utils.data.DataLoader,
               loss_fn,
               optimizer: torch.optim.Optimizer,
               device = 'cpu'):
    
    model.train()
    
    train_loss, train_acc = 0, 0
    
    #loop through dataloader
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y_logits = model(X)
        
        loss = loss_fn(y_logits, y)
        train_loss += loss
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        # Calc accuracy metric
        y_logits_class = torch.argmax(y_logits, dim=1)
        train_acc += (y_logits_class==y).sum().item()/len(y_logits)
        
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    print('dwadwada', train_acc)
    return train_acc, train_loss

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device='cpu'):
    
    model.eval()
    
    test_loss, test_acc = 0, 0
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            y_logits = model(X)
            
            loss = loss_fn(y_logits, y)
            test_loss += loss.item()
            
            y_pred_class = torch.argmax(y_logits, dim=1)
            test_acc += (y_pred_class==y).sum().item()/len(y_logits)
        
        # Avg loss + acc per batch
        
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc


### Train function to combine trainstep and teststep
from tqdm.auto import tqdm

def train(model,
          test_dataloader,
          train_dataloader,
          loss_fn,
          optimizer,
          epochs,
          device='cpu'):
    for epochs in tqdm(range(epochs)):
        train_acc, train_loss = train_step(model=model,
                   dataloader= train_dataloader,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   device=device)
        
        test_loss, test_acc = test_step(model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        print(f'Train Loss: {train_loss:.3f} | train_acc: {train_acc:.3f} | test loss: {test_loss:.3f} | test acc: {test_acc:.3f}')

# Timer
from timeit import default_timer as timer
start_Time = timer()


train(model=model_0,
      test_dataloader=test_batches,
      train_dataloader=train_batches,
      loss_fn=loss_func,
      optimizer=optimizer,
      epochs=5,
      device='cuda',)

end_Time = timer()
print(f"Training time: {end_Time-start_Time:.3f}")