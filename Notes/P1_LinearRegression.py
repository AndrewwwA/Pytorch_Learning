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
    
   
    # %% RUN CELL
    
    
    plot_predictions()
    
    # First Linear Regression MODEL =======
    # FORMULA ---- y = a + bx ----
    # class for linear regression model
    # What the model does is start with random values. looks at trainign data and adujust weight + bias to correlate to the correct data values 
    
    class LinearRegressionModel(nn.Module): # Pytorch comanly starts with nn.Module
        def __init__(self):
            # super().__init__()
            self.weight =  nn.Parameter(torch.randn(1,
                                                    requires_grad=True, 
                                                    dtype=torch.float))
            self.bias = nn.Parameter(torch.randn(1, 
                                                requires_grad=True, 
                                                dtype=torch.float))
            
            
            # Forward method
            def forward(self, x: torch.Tensor) -> torch.Tensor: 
                return self.weight * x + self.bias # Linear regression formula
        
    








