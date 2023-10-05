# Pytorch computer vision libraries
# %%
# Torch vision - base/main vision library for pytorch vision
# torchvision.datasets - get datasets and data loading functions for comp vision
# torchvision.models - pretrained computer vision models 
# torchvision.transforms - functions for manipulating vision data (images) to work for your model
# torch.utils.data.Dataset - base dataset class.
# torch.utils.data.Dataloader - creates a python iterable over a dataset

import torch
from torch import nn

# Torch vision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

# Getting a dataset (FashionMNIST) [torchvision.datasets]

### Setup data ###
train_data = datasets.FashionMNIST(
    root="data", #Where to download
    train=True, # Traiing or test dataset
    download=True, # Whether to download
    transform=torchvision.transforms.ToTensor(), #What transformer do we want to use on the dataset?
    target_transform=None #Do we want to transoform the labels/targets
)
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)
image, label = train_data[0]
# print(image.shape)
# torch.Size([1, 28, 28])

# Visualizing the image (Shows original image and label name)
# plt.imshow(image.squeeze(), cmap='gray')
# plt.title(train_data.classes[label])

# Plot 4x4 images (16 random images)
# fig = plt.figure(figsize=(9, 9))
# rows, cols = 4, 4
# for i in range(1, rows*cols + 1 ):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.axis(False)
#     plt.imshow(img.squeeze(), cmap='gray')
#     plt.title(train_data.classes[label])
    
### Prepare dataloader ###
# Data loaders turn the dataset into a more iterable Python "array"
# Breaking dataset of mini-matches for memory purposes (computationally efficient)
# Gives the NN to update it's gradients per epoch commonly in batch sizes of 32

from torch.utils.data import DataLoader
BATCH_SIZE = 32
train_batches = DataLoader(batch_size=BATCH_SIZE, 
                           shuffle=True, 
                           dataset=train_data)

test_batches = DataLoader(batch_size=BATCH_SIZE,
                          shuffle=False,
                          dataset=test_data)

train_features_batch, train_labels_batch = next(iter(train_batches))
print()
print(train_features_batch.shape, train_labels_batch.shape)

### Makign a baseline model ###
# Simple model to improve abpon with subsequent models/experiments 

# Flatten layer
flatten_model = nn.Flatten()

# Get a single sample
x = train_features_batch[0]
# print(x.shape)
# torch.Size([1, 28, 28])

# flatten sample -> [color_channels, height, width] => [color_channels, height*width]
output = flatten_model(x) # forward pass
# print(output.shape)
# torch.Size([1, 784])

class FashionMNISTModel(nn.Module):
    def __init__(self,
                 input_shape,
                 hidden_units,
                 output_shape):
        super().__init__()
        
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, 
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units, 
                      out_features=output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)
    
model_0 = FashionMNISTModel(input_shape=784, # 28 x 28 from flatten layer
                            hidden_units=10,
                            output_shape=10)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Loss function, optimizer, evaluation metrics
loss_func = nn.CrossEntropyLoss()

# optimizer
Optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# Evalution metrics
from helper_functions import accuracy_fn

# Function to time model
from timeit import default_timer as timer
def train_time(start: float,
               end: float,
               device: torch.device = None):
    total_time = end - start
    print(f"{total_time:.5f} seconds")
    
# EXAMPLE USE
# start_time = timer()
## Code
# end_time = timer()
# train_time(start=start_time, end=end_time, device='cpu')
# 0.0000 seconds

### Creating a training to training a model on batches of data and not EPOCHS ###
# Loop through epochs
# Loop through batches perform trianing steps calculate loss per batch
# Loop through testing with same steps
from tqdm.auto import tqdm

# Start timer
torch.manual_seed(42)
# start_time = timer()

# # Set epochs (currently small for testing)
# EPOCHS = 3

# # Training loop
# for epoch in tqdm(range(EPOCHS)):
#     print(f"Epoch: {epoch}\n-----")
    
#     #Training
#     train_loss = 0
    
#     # Loop to go through batches
#     for batch, (X, y) in enumerate(train_batches):
#         model_0.train()
        
#         # Forward pass
#         y_pred = model_0(X)
        
#         # Loss (per batch)
#         loss = loss_func(y_pred, y)
#         train_loss += loss # Train loss per EPOCH
        
#         # zero grad
#         Optimizer.zero_grad()
        
#         # back propagate
#         loss.backward()
        
#         # Grad descent
#         Optimizer.step()
        
#         # stats
#         if batch % 400 == 0:
#             print(f"on image {batch * len(X)}/{len(train_batches.dataset)}")
    
#     # Divide total train loss by length of train batch
#     train_loss /= len(train_batches)
    
#     ### Testing
#     test_loss, test_acc = 0, 0
#     model_0.eval()
#     with torch.inference_mode():
#         for X_test, y_test in test_batches:
#             # Forward pass
#             test_pred = model_0(X_test)
            
#             # Loss (With each batch)
#             test_loss += loss_func(test_pred, y_test)
            
#             # Accuracy
#             test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))
            
#         #  calculate test AVERAGE loss per batch
#         test_loss /= len(test_batches)
        
#         # Acc AVERAGE per batch
#         test_acc /= len(test_batches)
        
        
#     # Print data
#     print(f"\nTrain Loss: {train_loss:.4f} | Test_loss {test_loss:.4f}, | Test Acc: {test_acc:.4f}")
    
# # Training time
# end_time = timer()
# train_time(start_time, end_time)


# Make predictions and get MOdel 0 results
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader,
               loss_func,
               accuracy_func):
    """Returns a dictonary containg results of model's predictio"""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            # forward pass
            y_pred = model(X)
            
            # add loss per batch
            loss += loss_func(y_pred,
                              y)
            acc += accuracy_func(y,
                                 y_pred.argmax(dim=1))
        # Avg loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, #Returns if model created with calss
            "loss": loss.item(),
            "acc": acc}

# %%
# Calculate model 0 results on test dataset
# model_0_results = eval_model(model=model_0,
#                              data_loader=test_batches,
#                              loss_func=loss_func,
#                              accuracy_func=accuracy_fn)
# print(model_0_results)
# {'model_name': 'FashionMNISTModel', 'loss': 0.4766390025615692, 'acc': 83.42651757188499}

# Device agnostic code 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Model with non-lineararity and on GPU ###
class FashionMNISTModelV2(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_layers,
                 out_features):
        super().__init__()
        
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features,
                      out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers,
                      out_features=out_features),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)
model_1 = FashionMNISTModelV2(in_features=784,
                              hidden_layers=10,
                              out_features=len(train_data.classes)).to(device)
print(next(model_1.parameters()).device)
# cuda:0


### Loss function / Optimizer / training loop ###
loss_func = nn.CrossEntropyLoss()
# Optim
Optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)

### Function trainign and testing loops ###
def training_step(model,
                 dataloader,
                 loss_func,
                 optimizer,
                 accuracy_func,
                 device = device
                 ):
    train_loss, train_acc = 0, 0
    
    model.train()
    model.to(device)
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        
        loss = loss_func(y_pred, y)
        train_loss += loss
        train_acc += accuracy_func(y, y_pred.argmax(dim=1))
        
        optimizer.zero_grad()
        
        loss.backward()
        
        Optimizer.step()
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    print(f"Train loss: {train_loss:.5f} | train acc {train_acc:.3f}% ")

def testing_step(model,
                 dataloader,
                 loss_func,
                 accuracy_func,
                 device = device):
    test_loss, test_acc = 0, 0
    
    model.eval()
    
    with torch.inference_mode():
        for X_test, y_test in dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            y_pred = model(X_test)
            
            loss = loss_func(y_pred,
                             y_test)
            test_loss += loss
            test_acc += accuracy_func(y_test, 
                                      y_pred.argmax(dim=1))
            
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        
        print(f"loss: {loss:.5f} | accuracy: {test_acc:.3f}")


### TRAINING AND TESTING LOOP USIGN FUNCTIONS ON MODEL V1 ###
torch.manual_seed(42)

# Time 
start_time = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    training_step(model=model_1,
                  dataloader=test_batches,
                  loss_func=loss_func,
                  optimizer=Optimizer,
                  accuracy_func=accuracy_fn,
                  device=device)
    testing_step(model=model_1,
                 dataloader=test_batches,
                 loss_func=loss_func,
                 accuracy_func=accuracy_fn,
                 device=device)
end_time = timer()
train_time(start_time, end_time)

        
        
            
            
        
        
            



    
    
    



