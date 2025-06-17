import torch
# Question 5: Broadcasting Rules
def broadcasting_examples():
    """
    Create examples that demonstrate PyTorch broadcasting rules:
    1. (3,1) + (1,4) = (3,4)
    2. (2,3,1) * (1,1,4) = (2,3,4)
    3. Show a case that would fail broadcasting

    Learning: Broadcasting mechanics, common pitfalls

    When operating on two arrays, Pytorch compares their shapes element-wise. It starts with the trailing(left most) dimensions,
    and works its way forward. Two dimensions are compatible when :

    they are equal, or
    one of them is 1

    If these conditions are not met, an exception is thrown, indicating that the arrays have incompatible shapes.
    """
    x = torch.rand(3,1)
    y = torch.rand(1,4)
    print(f"The shape of x is {x.shape}")
    print(f"The tensor x is {x}")
    print(f"The shape of y is {y.shape}")
    print(f"The tensor y is {y}")
    z = x + y
    print(f"The shape of z is {z.shape}")
    print(f"The tensor z is {z}")

    a = torch.rand(2,3,1)
    b = torch.rand(1,1,4)
    print(f"The shape of a is {a.shape}")
    print(f"The tensor a is {a}")
    print(f"The shape of b is {b.shape}")
    print(f"The tensor b is {b}")
    c = a * b
    print(f"The shape of c is {c.shape}")
    print(f"The tensor c is {c}")

    k = torch.rand(3,2)
    n = torch.rand(3,3)
    m = k + n
    print(f"The shape of m is {m.shape}")
    print(f"The tensor m is {m}")
broadcasting_examples()
