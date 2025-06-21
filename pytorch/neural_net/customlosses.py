
import torch
import torch.nn.functional as F

# Question 11: Loss Functions
class CustomLosses:
    """
    Implement common loss functions from scratch.

    Learning: Loss function mathematics, reduction strategies
    """
    @staticmethod
    def mse_loss(predictions, targets, reduction='mean'):
      loss = torch.square(targets - predictions)
      if(reduction == 'mean'):
        loss = torch.mean(loss)
      elif(reduction == 'sum'):
        loss = torch.sum(loss)
      return loss

    @staticmethod
    def cross_entropy_loss(logits, targets):
      softmax = torch.exp(logits) / torch.sum(torch.exp(logits) , dim=1 , keepdim=True )
      loss = -torch.sum(targets * torch.log(softmax + 1e-10) , dim=1 )
      return torch.mean(loss)

    @staticmethod
    def binary_cross_entropy(predictions, targets):
      loss = -(targets * torch.log(predictions + 1e-10) + (1-targets) * torch.log(1-predictions + 1e-10))
      return torch.mean(loss)

loss = CustomLosses()
torch.manual_seed(42)
x = torch.rand(3 ,3)
y = torch.rand(3 ,3)
print(f"The output of custom mse loss is {loss.mse_loss(x , y)}")
print(f"The output of custom cross entropy loss is {loss.cross_entropy_loss(x , y)}")
print(f"The output of custom binary cross entropy loss is {loss.binary_cross_entropy(x , y)}")
print(f"The output of mse loss in pytorch is {F.mse_loss(x , y)}")
print(f"The output of cross entropy loss in pytorch is {F.cross_entropy(x , y)}")
print(f"The output of binary cross entropy loss in pytorch is {F.binary_cross_entropy(x , y)}")

# Check if the custom function loss functions are similar with torch in built loss functions

print(torch.allclose(loss.mse_loss(x , y) , F.mse_loss(x , y)))
print(torch.allclose(loss.cross_entropy_loss(x , y) , F.cross_entropy(x , y)))
print(torch.allclose(loss.binary_cross_entropy(x , y) , F.binary_cross_entropy(x , y)))
