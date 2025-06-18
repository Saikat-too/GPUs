import torch
import torch.nn as nn
# Question 9: Linear Layer Implementation

# Question 9: Linear Layer Implementation
class LinearLayer(nn.Module):
    """
    Implement a linear layer from scratch using nn.Parameter.
    Include proper initialization and forward pass.

    Learning: nn.Module structure, Parameter vs regular tensors
    """
    # Initializing weights and bias of the model
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features , out_features))
        self.bias   = nn.Parameter(torch.randn(out_features))
    # Forward Pass
    def forward(self, x):
        return x @ self.weights + self.bias

# Initializing a linear network using our custom Linear net then with pytorch Linear net
l1 = LinearLayer(3 , 5)
l2 = nn.Linear(3 , 5)

print(f"The weights of l1 is {l1.weights}")
print(f"The weights of l2 is {l2.weight}")
print(f"The bias of l1 is {l1.bias}")
print(f"The bias of l2 is {l2.bias}")

# Making sure that weights of the both network have same values so that we can
# compare the outputs  .
l1.weights.data.copy_(l2.weight.T)
l1.bias.data.copy_(l2.bias)


x = torch.randn(3, 3)
print(f"The input tensor is {x}")
print(f"The output of l1 is {l1(x)}")
print(f"The output of l2 is {l2(x)}")

assert(l1(x) == l2(x)).all()
