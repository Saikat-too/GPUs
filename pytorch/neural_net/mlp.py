import torch.nn as nn

# Question 12: Multi-Layer Perceptron
class MLP(nn.Module):
    """
    Create a configurable MLP with:
    - Variable number of hidden layers
    - Dropout
    - Batch normalization option
    - Different activation functions

    Learning: Model architecture design, regularization techniques
    """
    def __init__(self, input_size, hidden_sizes:list, output_size, dropout=0.2 , use_batchnorm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.layers = nn.ModuleList()
        current_size = self.input_size
        for i , hidden_size in enumerate (self.hidden_sizes):

            if i % 2 == 0:
              self.layers.append(nn.Linear(current_size , hidden_size))
              current_size = hidden_size
              self.layers.append(nn.ReLU())
              self.layers.append(nn.Dropout(self.dropout))
              self.dropout+=0.1

            if i % 2 == 1:
              self.layers.append(nn.Linear(current_size , hidden_size))
              current_size = hidden_size
              self.layers.append(nn.ReLU())
              if self.use_batchnorm:
                self.layers.append(nn.BatchNorm1d(hidden_size))


        self.layers.append(nn.Linear(current_size , self.output_size))
        self.layers.append(nn.Tanh())
    def forward(self, x):
        for layer in self.layers():
          x = layer(x)

        return x



net = MLP(256 , [128 , 64 , 32] , 8)
print(net)

'''

nn.ModuleList does not have a forward() method, because it does not define any neural network,
that is, there is no connection between each of the nn.Module's that it stores.
You may use it to store nn.Module's, just like you use Python lists to store other types of objects (integers, strings, etc).
The advantage of using nn.ModuleList's instead of using conventional Python lists to store nn.Module's is that Pytorch is
“aware” of the existence of the nn.Module's inside an nn.ModuleList, which is not the case for Python lists.

'''
