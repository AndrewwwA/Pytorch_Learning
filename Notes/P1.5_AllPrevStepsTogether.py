# %%
import torch 
from torch import nn
import matplotlib.pyplot as plt

# Device agnostic code (Uses gpu if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device: cuda

# Dummy data for models to try and figure out for y = weight * X + bias or y = mx + b FORMULA
weight = 0.9
bias = 0.05

# RAnge values
start = 0
end = 1
step = 0.02

# X and y (Features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1) # Without unsqueeze it'd be a matrix and not a tensor
y = weight * X + bias

# Training and test values
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# print(X[train_split:], y[train_split:])
# tensor([[0.8000],
#         [0.8200],
#         [0.8400],
#         [0.8600],
#         [0.8800],
#         [0.9000],
#         [0.9200],
#         [0.9400],
#         [0.9600],
#         [0.9800]]) 

# tensor([[0.7700],
#         [0.7880],
#         [0.8060],
#         [0.8240],
#         [0.8420],
#         [0.8600],
#         [0.8780],
#         [0.8960],
#         [0.9140],
#         [0.9320]])
# print(len(X_train), len(X_test))
# 40 10

# Plotting data (STOLE FROM P1)

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    '''
    Plot training data, tests data and compares predictions
    '''
    plt.figure(figsize=(10, 7))
    
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')
    
    # Plot test data in BROWN
    plt.scatter(test_data, test_labels, c='BROWN', s=4, label="Testing Data")
    
     # Show legends (Labels)
    plt.legend(prop={"size": 10})
    
    # Predictions if shown
    if predictions is not None:
        # PLOT PREDICTIONS IF TRUE (RED)
        plt.scatter(test_data, predictions, c='r', s=4, label="Predictions")


#  Building linear model by subclassing nn.Module (Not building params ourself but automatically done with nn using LAYERS)

class linearRegression2(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear() for creating the model params / linear layer / dense layer / probing layer / and more 
        # (1 Feature because the model takes in 1 value of x as the input and 1 out_features because 1 value of y as the output)
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
torch.manual_seed(40)
model_1 = linearRegression2()

# Setting device to Cuda hopefully
model_1.to(device)
# print(model_1.state_dict())
# OrderedDict([('linear_layer.weight', tensor([[-0.2642]], device='cuda:0')), ('linear_layer.bias', tensor([0.7322], device='cuda:0'))]) # Device = cuda

### Training ###

# Setting up Loss function and Optimizer
loss_Func = nn.L1Loss() # MAE 
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01, momentum=0.9)

# training loop
torch.manual_seed(40)

# After testing it is enough to reach target value close enough
epochs = 200

# Putting data on Cuda (Agnostic code)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test  = y_test.to(device)

for epoch in range(epochs):
    # Setting up model to train
    model_1.train()
     
    # Forward propagation
    y_pred = model_1(X_train)
   
    # Loss function
    loss = loss_Func(y_pred, y_train)
    
    # Zero grad
    optimizer.zero_grad()
    
    # backward propagation
    loss.backward()
    
    # Gradient descent (optimizer step)
    optimizer.step()
    
    # Testing    
    if epoch == 50:
        model_1.eval()

        with torch.inference_mode():
            test_predict = model_1(X_test)
            test_loss = loss_Func(test_predict, y_test)
        
        plot_predictions(predictions=test_predict.cpu())

   
        
    # if epoch % 10 == 0:
    #     print(f"Epoch: {epoch} | test_loss: {test_loss} | params: {model_1.state_dict()}")
     
     
### Making preditions ###

model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)

### Preditions visualization ### (Have to switch preditions because PLT cant use cuda only cpu) (Tensor.cpu())
plot_predictions(predictions=y_preds.cpu())

from pathlib import Path

### Saving / Loading models ###

# model directory
Model_Path = Path('models')
Model_Path.mkdir(parents=True, exist_ok=True)

# Create model save path
Model_Name = "01_Grad_Desc_2.pth"
Model_Save_Path = Model_Path / Model_Name

# Save model state dict
torch.save(model_1.state_dict(), Model_Save_Path)


        


