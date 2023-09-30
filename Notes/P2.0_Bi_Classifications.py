import sklearn
from sklearn.datasets import make_circles

# 1000 samples
n_samples = 1000

# create circles
X, y = make_circles(n_samples, 
                    noise=0.03,
                    random_state=42)
# print(X[:5], 'y', y[:5])

# Dataframe of circle dat using pandas
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})
circles.head(10) 

# Visualize with matplotlib
import matplotlib.pyplot as plt
# plt.scatter(x=X[:, 0],
#             y=X[:, 1],
#             c=y,
#             cmap=plt.cm.RdYlBu)

# Creating tensors from X and y values
# print(X.shape, y.shape)
# (1000, 2) (1000,)
import torch
# print(type(X)) # since X is a numpy array have to convert it to a torch arrary.
# <class 'numpy.ndarray'>
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
# print(X[:5], y[:5]) # They're automatically converted to a tensor datatype
# tensor([[ 0.7542,  0.2315],
#         [-0.7562,  0.1533],
#         [-0.8154,  0.1733],
#         [-0.3937,  0.6929],
#         [ 0.4422, -0.8967]])
# tensor([1., 1., 1., 1., 0.])

# Split data into trainign and test sets 
from sklearn.model_selection import train_test_split

X_train, X_test, y_trian, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # Splits 80% train and 20% test
                                                    random_state=42)


# OLD WAY TO SPLIT THE DATA
# split_data = int(0.8 * len(X))

# X_train, y_train = X[:split_data], y[:split_data]
# X_test, y_test = X[split_data:], y[:split_data]



### BUILDING A MODEL FOR CLASSIFICATIONS OF THE BLUE AND RED DOTS ###

# Agonistic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Constructing a model
# Creating 2 nn.Lienar layers to handle the two X values in the data
from torch import nn

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 nn.Lienar layers to handle the two X values in the data
        self.layer_1 = nn.Linear(in_features=2, # Takes in 2 features and outputs 5 features
                                 out_features=5)
        self.layer_2 = nn.Linear(in_features=5, # Takes in the 5 features and outputs a answer (same shape as y)
                                 out_features=1)
    # Forward method that outlines the forward pass
    def forward(self, x):
        return self.layer_2(self.layer_1(x)) # x -> layer_1 -> layer_2 -> answer (y)

# Creating a isntance of model and use gpu

model_0 = CircleModel().to(device)
# print(next(model_0.parameters()))
# tensor([[-0.6712,  0.6366],
#         [-0.3475, -0.3688],
#         [-0.5735, -0.6312],
#         [ 0.4626, -0.1784],
#         [-0.0715, -0.1020]], device='cuda:0', requires_grad=True)

# Replicate the model above using nn.Sequential() easier method to make a model with simple forward propagation method
model_02 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)
# print(model_02)
# Sequential(
#   (0): Linear(in_features=2, out_features=5, bias=True)
#   (1): Linear(in_features=5, out_features=1, bias=True)
# )
# print(model_02.state_dict())
# OrderedDict([('0.weight', tensor([ (Ordered as outfeatures output in featur es then outfeatrues [2 * 5 = 10, 5, 5, 1])
#         [ 0.3572, -0.0704],
#         [-0.6469,  0.2411],
#         [-0.4503,  0.6797],
#         [ 0.2890,  0.2931],
#         [-0.2983,  0.2746]], device='cuda:0')), 
#         ('0.bias', tensor([ 0.3393,  0.5001, -0.4974,  0.0819, -0.1017], device='cuda:0')), 
#         ('1.weight', tensor([[ 0.4471,  0.2877,  0.0184,  0.3033, -0.2020]], device='cuda:0')), 
#         ('1.bias', tensor([-0.4321], device='cuda:0'))])

### Setup loss fucntion and optimizer ###
#  Because it's a classification problem  L1Loss wouldn't work properly because it's for regression 
#  for classification we want binary cross entrpyy and categorical cross entropy 
#  for optimizers you want SGD and Adam 

loss_func = nn.BCELoss # Requires inputs to have gone through the sigmoid activation function function prior to int to BCELoss
loss_func = nn.BCEWithLogitsLoss() # Sigmoid activation function build in

optimizer = torch.optim.SGD(params=model_02.parameters(), lr=0.1)

# Calculate accuracy - out of 100 examples what percentage does our model get right?
def accuracy_func(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct/len(y_pred) * 100
    return acc

# Training classifciation model 
# Going from raw logits -> prediction probabilities -> prediction labels
# model outputs are going to be raw **Logits** you can convert them into probabilities by passing them into a kind of activation function 
# Exp: sigmoid for binary classification  and softmax for multiclass classification

# View the first 5 outputs of the forward pass on the test data
model_02.eval()
with torch.inference_mode():
    y_logits = model_02(X_test.to(device))[:5]
    # print(y_logits)
    # tensor([[-0.3619], # These are considered logits (the raw output before being put through a activation function)
    #        [-0.2527],
    #        [-0.5652],
    #        [-0.3046],
    #        [-0.4338]], device='cuda:0')

# Use the sigmoid activation function on the model's logits to turn them into prediction probabilities
y_pred_probs = torch.sigmoid((y_logits))
# print(y_pred_probs)
# tensor([[0.5733],
#         [0.5719],
#         [0.5675],
#         [0.5750],
#         [0.5520]], device='cuda:0')
# for your prediction prob if y > 0.5 = 1 else y = 0
y_preds = torch.round(y_pred_probs)
# print(torch.round(y_pred_probs))
# tensor([[1.],
#         [1.],
#         [1.],
#         [1.],
#         [1.]], device='cuda:0')

# In full (logits -> pred probs -> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_02(X_test.to(device))[:5]))

# Check for equality
# print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
# tensor([True, True, True, True, True], device='cuda:0')

# Get rid of extra dims
# print(y_preds.squeeze())
# tensor([1., 1., 1., 1., 1.], device='cuda:0')

### Training and testing loop ###
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100

# Setting data to device
X_train, y_train = X_train.to(device), y_trian.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(0): #currently changed to 1 for non waste of time
    model_02.train()
    
    # Forward pass
    y_logits = model_02(X_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits)) # Turn logits -> pred probs -> pred labels
    
    loss = loss_func(y_logits,    # Loss / accuracy # Using y_logits and not sigmoid(y_logits) because were using BCELossWithLogits and not BCELoss
                     y_train)
    
    accuracy = accuracy_func(y_preds, y_train)
    
    # Optimize zero grad
    optimizer.zero_grad()
    
    # Loss backwards
    loss.backward()
     
    # optimizer step
    optimizer.step()
    
    ### TESTING ###
    
    if epoch % 10 == 0:
        model_02.eval()
        
        with torch.inference_mode():
            # Forward pass
            test_logits = model_02(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            
            # Calculate test loss/acc
            test_loss = loss_func(test_logits,
                                y_test)
            test_acc = accuracy_func(test_pred, y_test)
        
        print(f"epoch: {epoch} | Acc: {test_acc} | loss: {loss:.5f}")

from helper_functions import plot_predictions, plot_decision_boundary

# see how the model puts the datapoints 
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Train')
# plot_decision_boundary(model_02, X_train, y_train)            

### Creating a model to deal with non-linear classification ### 
class CircleModelV3(nn.Module):
    def __init__(self):
        super().__init__()
        # Can be improved quicker drastically by increasing in and out features of hiddn layers
        self.layer1 = nn.Linear(in_features=2, out_features=20)
        self.layer2 = nn.Linear(in_features=20, out_features=20)
        self.layer3 = nn.Linear(in_features=20, out_features=1)
        self.relu = nn.ReLU() #Non-linear activation function
    
    def forward(self, x): # Uses a relu activation after every layer
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))
        
model_3 = CircleModelV3().to(device)

# Setup a new loss function to use MODEL_3 params now
loss_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)

### Training loop for new non-linear model ###

# Random seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

epochs = 1000 # Should recieve 100% on 2000 epoch with only 10 in and out features about 1200 on 20 features

for epoch in range(epochs):
    model_3.train()
    
    # Forward pass
    y_logits = model_3(X_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits)) # logits -> pred probabilities -> prediction labels
    
    # loss
    loss = loss_func(y_logits, y_train)
    accuracy = accuracy_func(y_preds, y_train)
    
    # zero grad
    optimizer.zero_grad()
    
    # back prop
    loss.backward()
    
    # gradient descent
    optimizer.step()
    
    if epoch % 100 == 0 or epoch == 1999:
        
        model_3.eval()
        
        with torch.inference_mode():
            test_logits = model_3(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_func(test_logits, y_test)
            test_acc = accuracy_func(test_pred, y_test)
            print(f' epoch: {epoch} | loss: {test_loss} | accuracy {test_acc}')

# Visualize the model trained with {epoch} epochs

model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()
    
# Plotting descision boundaries
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Train')
plot_decision_boundary(model_3, X_train, y_train)    


            

 
        
        
        
        
    
    
    






        
