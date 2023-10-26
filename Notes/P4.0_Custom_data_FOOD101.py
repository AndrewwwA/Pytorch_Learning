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
                         num_workers=os.cpu_count() - 1,
                         shuffle=True,
                         dataset=train_data,
)
test_batches = DataLoader(batch_size=1,
                        num_workers=os.cpu_count() - 1,
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
                                  num_workers=os.cpu_count() - 1,
                                  shuffle=True,
                                  dataset=simple_train_data)
simple_test_batches = DataLoader(batch_size=32,
                                 num_workers=os.cpu_count() - 1,
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
                padding=1,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
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
                padding=1,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
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
                in_features=hidden_units,
                out_features=3
            )
        )
        
    def forward(self, x):
        x = self.convLayer1(x)
        print(x)
        x = self.convLayer2(x)
        print(x)
        x = self.classifier(x)
        print(x)
        return x
        # reuturn self.classifier(self.convLayer2(self.convLayer1(x)))

torch.manual_seed(42)
model_0 = TinyVGG(input=3, # color channels
                  hidden_units=10,
                  output=3 # Num of options in classifier
                  ).to('cuda')
print(model_0)

