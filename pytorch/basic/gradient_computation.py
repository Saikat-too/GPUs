from IPython.core.display import Image , display
import torch
from torchviz import make_dot
# Question 3: Gradient Computation
def gradient_computation():
    """
    Create tensors x and y, compute z = x^2 + y^3, and find gradients.
    Include a case where you need to retain_graph=True

    Learning: Autograd, computational graphs, gradient retention
    """
    x = torch.tensor([5.0] , requires_grad=True)
    y = torch.tensor([6.0] , requires_grad=True)
    z = x**2 + y**3
    dot = make_dot(z , params={'x':x , 'y':y})
    dot.render("computation_graph", format="png",cleanup=True)
    display(Image(filename="computation_graph.png"))
    print(f"The value of z is {z.item()}")
    z.backward(retain_graph=True)
    print(f"The x gradient is {x.grad.item()}")
    print(f"The y gradient is {y.grad.item()}")

    # x.grad = None
    # y.grad = None

    z.backward()

    print(f"The x gradient is {x.grad.item()}")
    print(f"The y gradient is {y.grad.item()}")



    '''
   To reduce memory usage, during the .backward() call, all the intermediary results are deleted when they are not needed anymore.
    Hence if you try to call .backward() again, the intermediary results don’t exist and the backward pass cannot be performed (and you get the error you see).
    You can call .backward(retain_graph=True) to make a backward pass that will not delete intermediary results,
    and so you will be able to call .backward() again. All but the last call to backward should have the retain_graph=True option.

    '''

gradient_computation()
