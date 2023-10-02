import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy
from torch import nn

RANDOM_SEED = 42
SAMPLES = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

### 1. Make a binary classification dataset with Scikit-Learn's make_moons() function.

# For consistency, the dataset should have 1000 samples and a random_state=42.
X, y = make_moons(n_samples=SAMPLES, random_state=RANDOM_SEED, noise=0.07)
# print(X_moon_train.shape, y_moon_train.shape)
# (1000, 2) (1000,)
# Turn the data into PyTorch tensors.
X, y = torch.Tensor(X).type(torch.float), torch.Tensor(y).type(torch.float)
# Split the data into training and test sets using train_test_split with 80% training and 20% testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_SEED, test_size=0.2)

# print(len(X_train), len(X_test), len(y_test), len(y_train))
# 800 200 200 800

# 2. Build a model by subclassing nn.Module that incorporates non-linear activation functions and is capable of fitting the data you created in 1.
# Feel free to use any combination of PyTorch layers (linear and non-linear) you want.

class ModelMultiClass(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )
    def forward(self, x):
        return self.layers(x)

model_0 = ModelMultiClass().to(device)

# 3. Setup a binary classification compatible loss function and optimizer to use when training the model built in 2.
loss_func = nn.BCEWithLogitsLoss()
# optimizer
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)

# 4. Create a training and testing loop to fit the model you created in 2 to the data you created in 1.
# To measure model accuray, you can create your own accuracy function or use the accuracy function in TorchMetrics.
# Train the model for long enough for it to reach over 96% accuracy.
# The training loop should output progress every 10 epochs of the model's training and test set loss and accuracy.
acc_func = Accuracy(task="multiclass", num_classes=2).to(device) # send accuracy function to device

torch.manual_seed(42)
epochs = 1 # Changed to 1 for performance but 1000 fits all requirements

X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()

    logits = model_0(X_train).squeeze()
    pred_labels = torch.round(torch.sigmoid(logits))  

    loss = loss_func(logits, y_train)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()    
    
    if epoch % 100 == 0 or epoch == 999:
        model_0.eval()
        with torch.inference_mode():
            test_logits = model_0(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            
            test_loss = loss_func(test_logits, y_test)
            test_acc = acc_func(test_pred, y_test)
            
            # print(f"epoch: {epoch} | test_acc: {test_acc} | loss: {test_loss}")
            # epoch: 999 | test_acc: 1.0 | loss: 0.014763526618480682
            

     
import numpy as np

def plot_decision_boundary(model, X, y):
  
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Source - https://madewithml.com/courses/foundations/neural-networks/ 
    # (with modifications)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), 
                         np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # mutli-class
    else: 
        y_pred = torch.round(torch.sigmoid(y_logits)) # binary
    
    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# # Plot decision boundaries for training and test sets

### UNREMOVE FOR DATA ###
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_0, X_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_0, X_test, y_test)
     
     
# 6. Replicate the Tanh (hyperbolic tangent) activation function in pure PyTorch.
# Feel free to reference the ML cheatsheet website for the formula.

def tanh(z):
	return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

# # Create a straight line tensor
str_line = torch.arange(-100, 100, 1)
# print(str_line)
# tensor([0.0000, 0.0500, 0.1000, 0.1500, 0.2000, 0.2500, 0.3000, 0.3500, 0.4000,
#         0.4500, 0.5000, 0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500,
#         0.9000, 0.9500])

### UNCOMMENT TO SEE
# plt.plot(tanh(str_line))


# 7. Create a multi-class dataset using the spirals data creation function from CS231n (see below for the code).
# Split the data into training and test sets (80% train, 20% test) as well as turn it into PyTorch tensors.
# Construct a model capable of fitting the data (you may need a combination of linear and non-linear layers).
# Build a loss function and optimizer capable of handling multi-class data (optional extension: use the Adam optimizer instead of SGD, you may have to experiment with different values of the learning rate to get it working).
# Make a training and testing loop for the multi-class data and train a model on it to reach over 95% testing accuracy (you can use any accuracy measuring function here that you like) - 1000 epochs should be plenty.
# Plot the decision boundaries on the spirals dataset from your model predictions, the plot_decision_boundary() function should work for this dataset too.

# # Code for creating a spiral dataset from CS231n
import numpy as np
import matplotlib.pyplot as plt
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
  
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
# plt.show()
     


# # Turn data into tensors
X = torch.from_numpy(X).type(torch.float) # features as float32
y = torch.from_numpy(y).type(torch.LongTensor) # labels need to be of type long

# # Create train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# print(len(X_train), len(X_test), len(y_train), len(y_test))
# 240 60 240 60

     
acc_fn = Accuracy(task="multiclass", num_classes=4).to(device)
     

# # Create model by subclassing nn.Module
class MutliClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=3)
        )    
    def forward(self, x):
        return self.layers(x)

# Instantiate model and send it to device
model_1 = MutliClassModel().to(device)

# # Setup data to be device agnostic
X_test, X_train = X_test.to(device), X_train.to(device)
y_test, y_train = y_test.to(device), y_train.to(device)

pre_logits = model_1(X_train)
     
# # Setup loss function and optimizer
loss_func_multi = nn.CrossEntropyLoss()
optimizer_multi = torch.optim.Adam(params=model_1.parameters(), lr=0.02)
     
# # Build a training loop for the model

epochs_multi = 1000

for epoch in range(epochs_multi):
    model_1.train()
    # 1. Forward pass
    mul_logits = model_1(X_train)
    mul_prob = torch.softmax(mul_logits, dim=1)
    mul_labels = torch.argmax(mul_prob, dim=1)

    # 2. Calculate the loss
    mul_loss = loss_func_multi(mul_logits, y_train)
    
    # 3. Optimizer zero grad
    optimizer_multi.zero_grad()
    
    # 4. Loss backward
    mul_loss.backward()

    # 5. Optimizer step
    optimizer_multi.step()

    ## Testing
    if epoch % 100 == 0 or epoch == 1999:
        model_1.eval()
        logits_mul_test = model_1(X_test)
        mul_labels_test = torch.argmax(torch.softmax(logits_mul_test, dim=1), dim=1)
        
        loss_mul_test = loss_func_multi(logits_mul_test, y_test)
        acc_test = acc_fn(mul_labels_test, y_test)
        
        print(f"Epoch: {epoch} | loss: {loss_mul_test} | acc: {acc_test}")
    
  

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)

