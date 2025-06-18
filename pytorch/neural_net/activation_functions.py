import torch
import torch.nn.functional as F

# Question 10: Activation Functions
class ActivationFunctions:
    """
    Implement various activation functions from scratch and compare with PyTorch versions.

    Learning: Activation function mathematics, numerical stability
    """
    @staticmethod
    def custom_relu(x):
        relu = torch.clamp(x , min = 0)
        return relu

    @staticmethod
    def custom_sigmoid(x):
        sig = 1 / (1 + torch.exp(-x))
        return sig

    @staticmethod
    def custom_tanh(x):
        tnh = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        return tnh


    @staticmethod
    def leaky_relu(x, negative_slope=0.01):
        l_relu = torch.where(x > 0 , x , negative_slope * x)
        return l_relu

activation = ActivationFunctions()
x = torch.randn(3 ,3)
y = x.clone().detach()
print(f"The output of custom relu is {activation.custom_relu(x)}")
print(f"The output of custom sigmoid is {activation.custom_sigmoid(x)}")
print(f"The output of custom tanh is {activation.custom_tanh(x)}")
print(f"The output of leaky relu is {activation.leaky_relu(x)}")

print(f"The output of torch relu is {F.relu(y)}")
print(f"The output of torch sigmoid is {F.sigmoid(y)}")
print(f"The output of torch tanh is {F.tanh(y)}")
print(f"The output of torch leaky relu is {F.leaky_relu(y)}")

print(torch.allclose(activation.custom_relu(x) , F.relu(y)))
print(torch.allclose(activation.custom_sigmoid(x) , F.sigmoid(y)))
print(torch.allclose(activation.custom_tanh(x) , F.tanh(y)))
print(torch.allclose(activation.leaky_relu(x) , F.leaky_relu(y)))
