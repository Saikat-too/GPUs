import torch
# Question 6: Tensor Memory Layout
def memory_layout():
    """
    Create tensors with different memory layouts (contiguous vs non-contiguous).
    Use .view() vs .reshape() appropriately.

    Learning: Memory efficiency, when to use contiguous()
    """

    # When data is contiguous
    x = torch.arange(1 , 11)
    print(f"The tensor x is {x}")

    # Reshape returns a view with new dimension when (when data is contiguous reshape() works like view())
    y = x.reshape(2 , 5)
    print(f"The new tensor y of reshaping x to (2,5) dimension is {y}")

    # How do we know it's a view ? Because element changes in y will effect in the x and vice versa .
    y[0][0] = 15
    print(f"Ater changing an element in y the tensor y is {y}")
    print(f"After changing an element in y the tensor x is {x}")

    # After transpose ,  data is not contiguous
    t = torch.arange(1 , 13).view(2 , 6).transpose(0 , 1)
    print(f"The tensor t is {t}")

    # Reshape() works fine on a non contiguous data
    z = t.reshape(4 , 3)
    print(f"The tensor t after reshaping {z}")

    # Change an element in z
    z[0][0] = 15
    print(f"After changing an element in z the tensor z is {z}")
    print(f"After changing an element in z the tensor t is {t}")

    # Let's see if view works in non contiguous data
    k = t.view(4 , 3)
    print(f"After doing view operation in t the tensor is {k}")





memory_layout()
