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
#  Loss/cost function: A function to measure of off your models predictions are to the ideal outputs (Lower value = better predictions)
#  Optimizer: Takes into account the loss value of a model and adjuts the model's parameters (CURRENTLY Weight and Bias) [SEEN BELOW]
#  print(model_0.state_dict()) : OrderedDict([('weight', tensor([0.3367])), ('bias', tensor([0.1288]))])
#  Using nn.L1Loss function for the loss problem. Creates a criterion that measures the mean absolute error (MAE)

#  Creating a loss/cost function for the model
loss_funcn = nn.L1Loss()

#  Creating a optimizer for the model
#  Using SGD stochastic (random) gradient descent 
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01, momentum=0.6)

#  Building the trianing and testing loop for the model
#  1. Forward Propagation (Using the models forward function)
#  2. Calculate the loss (comparing forward propagation predictions to the actual data's answer)
#  3. Optimizer zero grad
#  4. Backpropagation
#  5. Optimizer (Gradient descent)

# %%
# Each "Epoch" is a term for a the amount of times you loop through the data
torch.manual_seed(42)
epochs = 100

### ==== TRAINING === ###
# Looping throug the data
for epoch in range(epochs):
    # Setting the model to training mode
    model_0.train() # Set's all parametgers that require graident to true
    
    # Forward Pass
    y_pred = model_0(X_train)
    
    # Calculate loss
    loss = loss_funcn(y_pred, y_train)
    # Can view loss value each iteration
    # print('loss value', {loss})
    
    # Optimizer zero grad
    # Sets optimizer to zero each time to avoid conflicts 
    optimizer.zero_grad()
    
    # Backpropagation on loss with respect to parameters of the model to figure out gradient descent of every node
    loss.backward()
    
    # Perform gradient descent
    optimizer.step()
    
    # Turns off important things (Im no sure what) when testing your model (speeds up the testing process)
    model_0.eval()
    
    # === TESTING === PUT IT ALL IN THE SAME FUNCTON FOR STARTING GENRALLY THEY ARE IN FUNCTIONS
    # Turns off gradient tracking + more 
    with torch.inference_mode():
        # Forward propagation:
        test_prop = model_0(X_test)
        
        # Calculate cost/loss
        test_cost = loss_funcn(test_prop, y_test)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | 'info': {model_0.state_dict()} | test loss: {test_cost}")
    if epoch == 90:  # Can change depending on EPOCH AMOUNT TO HAVE A FINAL GRAPH (Currently set for 100  EPOCHS)
        print(plot_predictions(predictions=test_prop))

        
### Saving the model ###
# 1: torch.save() - allows you to save a pytorch object in python's pickle format (From python)
# 2: torch.load() - allows you to load a saved pytorch object 
# 3: torch.nn.Module.load_state_dict() = allows you to load a model's saved state dict

# # Saving this model
from pathlib import Path

# # Creating model directory
MODEL_PATH = Path('models')
# MODEL_PATH.mkdir(parents=True, exist_ok=True)

# # Create model save path
Model_NAME = "01_Grad_Desc.pth"
MODEL_SAVE_PATH = MODEL_PATH / Model_NAME

# print(MODEL_SAVE_PATH)
# # models\01_Grad_Desc.pth

# # Saving model State Dict (Models file/directory)
# torch.save(model_0.state_dict(), MODEL_SAVE_PATH)


### ========= LOADING MODELS ====== ###

# model_loaded = LinearRegressionModel()
# model_loaded.load_state_dict(torch.load(MODEL_SAVE_PATH))

# #  YOu can see that the model saved it's previous good Weight and bias from before

# print(model_loaded.state_dict())
# OrderedDict([('weight', tensor([0.7014])), ('bias', tensor([0.3019]))])

    
    
    












# %%
