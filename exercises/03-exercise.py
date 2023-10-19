# %%
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
    shuffle=False
)

train_features_batch, train_labels_batch = next(iter(train_batches))
# print(len(train_features_batch))
# 32
torch.manual_seed(42)
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
            nn.Linear(in_features=1960,
                      out_features=len(class_names)),
        )
    def forward(self, x):
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.SMBlock3(x)
        # print(x.shape)
        return x

torch.manual_seed(42)
Model_1 = MNISTTinyVGG(hidden_channels=40)
torch.manual_seed(42)
# Model_1(train_features_batch)
# torch.Size([32, 10, 14, 14])
# torch.Size([32, 10, 7, 7])
# torch.Size([32, 10])


### optimizer and loss ###
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=Model_1.parameters(),
                            lr=0.001)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


from timeit import default_timer as timer

def train_time(start: float,
               end: float):
    total_time = end - start
    print(f"{total_time:.5f} seconds")
    

def train_loop(model,
            loss,
            optimizer,
            dataset,
            device
            ):
    model = model.to(device)
    
    total_loss = 0
    
    for batch, (X, y) in enumerate(dataset):
        X, y = X.to(device), y.to(device)
        
        model.train()
        
        y_logit = model(X)
        
        loss = loss_func(y_logit, y)
        total_loss += loss
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        if batch % 470 == 0 or batch == 1874:
            print(f"Batch: {batch} / {len(dataset) - 1} | {loss}")
# %% 
start_time = timer()

epochs = 3
torch.manual_seed(42)
for epoch in range(epochs):
    train_loop(model=Model_1,
            loss=loss_func,
            optimizer=optimizer,
            dataset=train_batches,
            device='cuda')


end_time = timer()
print(f"Training Time: {train_time(start_time, end_time)}")
# 27.71460 seconds on GPU (FOR ME)
# 46.65879 seconds on CPU (FOR ME)

# %%
def test_loop(model,
            dataset,
            device):
    model.eval()
    
    model = model.to(device)
    with torch.inference_mode():
        total_acc = 0
        for X, y in dataset:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            y_pred = torch.argmax(y_logits, dim=1)
            
            total_acc += accuracy_fn(y, y_pred)
        print(f"Accuracy: {total_acc / len(dataset)}")

test_loop(model=Model_1,
           dataset=test_batches,
           device='cuda')

# %%
import torchmetrics, mlxtend

# Redo predictions
from tqdm.auto import tqdm
torch.manual_seed(42)
Model_1.eval()
Model_1.to('cuda')
guesses = []
with torch.inference_mode():
    for batch, (X, y) in tqdm(enumerate(test_batches)):
        X, y = X.to('cuda'), y.to('cuda')
        y_logits = Model_1(X)
        
        y_pred = torch.argmax(y_logits, dim=1)
        guesses.append(y_pred)
        
    guesses=torch.cat(guesses).cpu()
# print(len(guesses))
# 10000
print(test_data.targets[:10], guesses[:10])

# Plot in confusion matrix
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# Setup confusion matrix 
confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
confmat_tensor = confmat(preds=guesses,
                         target=test_data.targets)

# Plot the confusion matrix
plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(9, 6)
)


            
            
            
            
            
    

        
            
            
        
    
    
# %%
