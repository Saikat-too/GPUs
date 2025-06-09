
import torch
# Question 1: Basic Tensor Operations
def tensor_basics():
    """
    Create a 3x4 tensor with random values, then:
    1. Add 5 to all elements
    2. Multiply by 2
    3. Take the square root
    4. Calculate the mean across dimension 1

    Learning: Tensor creation, broadcasting, reduction operations

    """
    # Creating a 3 x 4 tensors with random values
    tensor = torch.rand(3 , 4)
    print(f'The tensor is {tensor}')

    # Adding 5 to all elements
    tensor = torch.add(tensor , 5)
    print(f'The tensor after adding 5 is {tensor}')

    # Multiply by 2
    tensor = torch.mul(tensor , 2)
    print(f'The tensor after multiplying by 2 is {tensor}')

    # Take the square root
    tensor = torch.sqrt(tensor)
    print(f'The tensor after taking the square root is {tensor}')

    # Mean across dimesion 1
    tensor = torch.mean(tensor , 1 , True)
    print(f'The tenosor after the mean across dimension 1 is {tensor}')
tensor_basics()
