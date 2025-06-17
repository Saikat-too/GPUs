import torch
# Question 4: In-place vs Out-of-place Operations
def inplace_operations():
    """
    Demonstrate the difference between in-place and out-of-place operations.
    Show what happens to gradients with in-place operations.

    Learning: Memory efficiency, gradient computation issues with in-place ops
    """

    # When we modify a tenosr using in-place operations it's memory doesn't change
    tens = torch.tensor([2.0], requires_grad=True)
    print(f"The memory location of the tens before any ops is  {id(tens)}")

    tens = tens + 2
    print(f"The memory location of the tens after out-place ops is {id(tens)}")

    tens += 2
    print(f"The memory location fo the tens after in-place  ops is {id(tens)}")

    # Doing gradient computation with out-place operations
    x_out = torch.tensor([5.0], requires_grad=True)
    y_out = torch.tensor([6.0], requires_grad=True)
    x_out = x_out * y_out
    x_out.backward()
    # Access gradients from the original leaf tensors
    print(f"After out-place operation backward x_out.grad is {x_out.grad}")
    print(f"After out-place operation backward y_out.grad is {y_out.grad}")

    # Doing gradient computation with in-place operation
    x_in = torch.tensor([5.0], requires_grad=True)
    y_in = torch.tensor([6.0], requires_grad=True)
    x_in *=y_in
    x_in.backward()
    # Access gradients from the original leaf tensors
    print(f"After inplace operation backward x_in.grad is {x_in.grad}")
    print(f"After inplace operation backward y_in.grad is {y_in.grad}")


inplace_operations()
