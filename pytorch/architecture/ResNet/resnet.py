import  torch.nn as nn
# Question 18: Residual Connections
class ResidualBlock(nn.Module):
    """
    Implement a residual block with optional downsampling.

    Learning: Skip connections, gradient flow, identity mapping
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Firstly implement the convolutional layer
        self.conv1 = nn.Conv2d(in_channels , out_channels , kernel_size=3 , stride=stride, padding=1 , bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels , out_channels , kernel_size=3 , stride=1 , padding=1 , bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Adding Skip connection or shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
          self.shortcut = nn.Sequential(
              nn.Conv2d(in_channels , out_channels , kernel_size=3 , stride=stride , padding=1 , bias=False),
              nn.BatchNorm2d(out_channels)
          )

    def forward(self, x):

       # Store the data in identity
       identity = x

       x = self.conv1(x)
       x = self.bn1(x)
       x = self.activation(x)

       x = self.conv2(x)
       x = self.bn2(x)

       # In place operation that's why we set Inplace = True in ReLU activation
       x+=self.shortcut(identity)
       x = self.relu(x)

       return x


model = ResidualBlock(3 , 64 )
print(model)
