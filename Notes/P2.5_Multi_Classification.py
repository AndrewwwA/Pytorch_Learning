# Multi-class classification problem 
# Used for classifying more then 2 things (Binary classification: Dogs vs Cats) (Multi-class classification: Dogs vs Cats vs Ferrets)
# Uses softmax instead of Sigmoid and uses CrossEntropyLoss instead of BCELoss
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

### Creating a "Toy" multi-class dataset ###


# Creating data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=2,
                            centers=4,
                            cluster_std=1.5, #mixes up with standard deviation
                            random_state=42)

# print(X_blob[:10], y_blob[:10])
# [[-8.41339595  6.93516545]
#  [-5.76648413 -6.43117072]
#  [-6.04213682 -6.76610151]
#  [ 3.95083749  0.69840313]
#  [ 4.25049181 -0.28154475]
#  [-6.7941075  -6.47722117]
#  [ 5.21017561  2.8890483 ]
#  [-5.50513568 -6.36037688]
#  [-6.54375599 -4.83389895]
#  [ 6.44964229  0.74776618]] [3 2 2 1 1 2 1 2 2 1]

# Turning data into Torch Tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.long) # Requires long for CrossEntropy

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_blob,
                                                    y_blob,
                                                    test_size=0.2,
                                                    random_state=42)
# Show data (plot)
plt.figure(figsize=(12, 6))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)

# Agnostic code
device = 'cuda' if torch.cuda.is_available else 'cpu'

#=========== ### Build model ### ===========#
class blob_Model(nn.Module):
    def __init__(self, in_features=2, out_features=4, hidden_units=8):
        """
        in_features: (Int) Number of input features (default=2)
        output_features: (Int) Number of outputs the model gives (default=4)
        hidden_units: (Int) Number of hidden units between layers (Default 8)
        
        """
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features)
        )
        
    def forward(self, x):
        return self.layer_stack(x)

# Instance 
torch.manual_seed(42)
model_0 = blob_Model(2, 4, 8).to(device)
    
# Loss & Optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.001)\
    
# Tests and training on GPU
X_test, y_test = X_test.to(device), y_test.to(device)
X_train, y_train = X_train.to(device), y_train.to(device)

# helper % correct
def accuracy_func(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct/len(y_pred) * 100
    return acc


### Training loop ###
torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    model_0.train()
    
    #Forward pass
    y_logits = model_0(X_train)
    
    #loss
    
    loss = loss_func(y_logits, y_train)
    
    # zero grad
    optimizer.zero_grad
    
    #back prob
    loss.backward()
    
    # gradient descent
    optimizer.step()
    
    if epoch % 10 == 0 or epoch == 99:
        model_0.eval()
        with torch.inference_mode():
            # forward pass
            y_logits = model_0(X_test)
            y_guess = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            
            y_acc = accuracy_func(y_guess, y_test)
            
            print(f"Epoch: {epoch} | loss: {loss} | accuracy: {y_acc}")

# Make predictions
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test)
    
    y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

# plotting
from helper_functions import plot_decision_boundary
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_test, y_pred)          

# Other ways to evalute the model (Use torchmetrics to use these quickly)
#  Accuracy - out of 100 samples, how many does the model get correct
# Precision
# Recall
# F1-score
# Confusion matrix
# Classification report

### Saving this model ###
# from pathlib import Path

# MODEL_PATH = Path('models')

# Model_NAME = "02_MultiClass_Classification.pth"
# MODEL_SAVE_PATH = MODEL_PATH / Model_NAME

# torch.save(model_0.state_dict(), MODEL_SAVE_PATH)

            

