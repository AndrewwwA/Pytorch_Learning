import torch
from torch import nn
import matplotlib.pyplot as plt

# 1. Create a straight line dataset using the linear regression formula (weight * X + bias).

# Set weight=0.3 and bias=0.9 there should be at least 100 datapoints total.
weight = 0.3
bias = 0.9
step = 0.01

X = torch.arange(0, 1, step).unsqueeze(1)
y = weight * X + bias
train_split = int(0.8 * len(X))

# Split the data into 80% training, 20% testing.
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Plot the training and testing data so it becomes visual.
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
        
plot_predictions()

# 2: Build a PyTorch model by subclassing nn.Module.
class linearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
# Inside should be a randomly initialized nn.Parameter() with requires_grad=True, one for weights and one for bias.
        self.linear_layer = nn.Linear(in_features=1,
                                    out_features=1)
# Implement the forward() method to compute the linear regression function you used to create the dataset in 1.        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(X)
# Once you've constructed the model, make an instance of it and check its state_dict().

torch.manual_seed(42)
model_0_exercise = linearRegressionModel()
# print(model_0_exercise.state_dict())
# OrderedDict([('linear_layer.weight', tensor([[0.7645]])), ('linear_layer.bias', tensor([0.8300]))])
       
# 4. Create a loss function and optimizer using nn.L1Loss() and torch.optim.SGD(params, lr) respectively.
loss_fn = nn.L1Loss()
# Set the learning rate of the optimizer to be 0.01 and the parameters to optimize should be the model parameters from the model you created in 2.
optimizer = torch.optim.SGD(params=model_0_exercise.parameters(), lr=0.01)
# Write a training loop to perform the appropriate training steps for 300 epochs.
torch.manual_seed(42)
Epochs = 300

for epoch in range(Epochs):
    model_0_exercise.train()
    
    # Forward prop
    y_pred = model_0_exercise(X_train)
    
    # Loss 
    loss = loss_fn(y_pred, y_train)
    
    # zero grad
    optimizer.zero_grad()
    
    # loss backwards
    loss.backward()
    
    # optimizer step
    optimizer.step()
# The training loop should test the model on the test dataset every 20 epochs. 
    if epoch % 20 == 0:
        with torch.inference_mode():
            print(f"loss: {loss} | params: {model_0_exercise.state_dict()}")
            
            
# 4. Make predictions with the trained model on the test data.
# Visualize these predictions against the original training and testing data

with torch.inference_mode():
    model_0_exercise.eval()
    y_pred_trained = model_0_exercise(X_test)
    
    plot_predictions(predictions=y_pred_trained)
        
    





