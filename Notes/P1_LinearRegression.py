# %%
# MODEL PREPERATION (DATA), BUILDING, TRAINING, PREDICTIONS, SAVING/LOADING MODEL, COMBINING

import torch
from torch import nn # nn is PyTorch Building modules for nueral networks
import matplotlib.pyplot as plt


# print(torch.__version__)

# DATA (PREPARING AND LOADING) ---
# DATA CAN BE ANYTHING... Excel, Images, Videos, Audio, Text, and more.
# ============ Making a straight line with know parameters using linear regression ==================

# Parameters =====
weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(1)
y = weight * X + bias

# print(X[:10])
# print(y[:10])

### Splitting data into trainign and test sets 
# print(len(X))
# 50



# USES 40 EXAMPLES TO GUESS THE LAST 10 EXAMPLES
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# print(len(X_train), len(y_train), len(X_test), len(y_test))
# 40 40 10 10   ### 40 X AND Y VALUES then 10 X AND Y TEST VALUES
 
 
# ---- MATPLOTLIB (TO VISUALIZE DATA --------------------


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
    
   
    # % RUN CELL
    
    
    # plot_predictions()
    
    # First Linear Regression MODEL =======
    # FORMULA ---- y = a + bx ----
    # class for linear regression model
    # What the model does is start with random values. looks at trainign data and adujust weight + bias to correlate to the correct data values 
    
class LinearRegressionModel(nn.Module): # Pytorch comanly starts with nn.Module
    def __init__(self):
        super().__init__()
        self.weight =  nn.Parameter(torch.randn(1,
                                                requires_grad=True, 
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, 
                                            requires_grad=True, 
                                            dtype=torch.float))
        
        
        # Forward method
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.weight * x + self.bias # Linear regression formula
        
# Check data inside model using .parameters() -------
torch.manual_seed(42)

# Creating instance of the model (subclass (LinearRegressionModel))
model_0 = LinearRegressionModel()

# print(model_0.state_dict())
# tensor([0.3367], requires_grad=True), Parameter containing:
# tensor([0.1288], requires_grad=True)]

# -----  Making predictions with random numbers using 'torch.inference_mode()' -------

# print(X_test)

# TURNS OFF GRADIENT TRACKING USING INFERENCE MODE (SPEEDS UP TESTING) -------------
with torch.inference_mode():
    y_prediction = model_0(X_test)

# print(y_prediction, 'correct', y_test)
# INCORRECT:  tensor([[0.3982],
#                     [0.4049],
#                     [0.4116],
#                     [0.4184],
#                     [0.4251],
#                     [0.4318],
#                     [0.4386],
#                     [0.4453],
#                     [0.4520],
#                     [0.4588]])

# % VERY FAR OFF ADD % TO RUN CELL
# plot_predictions(predictions=y_prediction)



### ===================== TRAINING THE MODEL ============================= ###
#  Measure how off your predictions are to the correct data is done by using a loss function (Commonly called a cost function or criterion depedning on the field).
#  Loss function: A function to measure of off your models predictions are to the ideal outputs (Lower value = better predictions)
#  Optimizer: Takes into account the loss value of a model and adjuts the model's parameters (CURRENTLY Weight and Bias) [SEEN BELOW]
#  print(model_0.state_dict()) : OrderedDict([('weight', tensor([0.3367])), ('bias', tensor([0.1288]))])
#  Using nn.L1Loss function for the loss problem. Creates a criterion that measures the mean absolute error (MAE)

#  Creating a loss function for the model
loss_funcn = nn.L1Loss()

#  Creating a optimizer for the model
#  Using SGD stochastic (random) gradient descent 
optimizer = torch.optim.SGD(params=model_0.paramters(), lr=0.01)

#  Building the trianing and testing loop for the model
#  1. Forward Propagation (Using the models forward function)
#  2. Calculate the loss (comparing forward propagation predictions to the actual data's answer)
#  3. Optimizer zero grad
#  4. Backpropagation
#  5. Optimizer (Gradient descent)


# Each "Epoch" is a term for a the amount of times you loop through the data
epochs = 1

### ==== TRAINING === ###
# Looping throug the data
for epoch in range(epochs):
    # Setting the model to training mode
    model_0.train() # Set's all parametgers that require graident to true
    
    # Forward Pass
    y_pred = model_0(X_train)
    
    # Calculate loss
    loss = loss_funcn(y_pred, y_train)
    
    # Optimizer zero grad
    # Sets optimizer to zero each time to avoid conflicts 
    optimizer.zero_grad()
    
    # Backpropagation on loss with respect to parameters of the model
    loss.backward()
    
    # Perform gradient descent
    optimizer.step()
    
    
    












# %%
