import torch
import torch.nn as nn

# Question 17: Convolutional Neural Network
class CNN(nn.Module):
    """
    Build a CNN for image classification with:
    - Multiple conv layers with pooling
    - Batch normalization
    - Global average pooling

    Learning: CNN architecture, feature extraction
    """
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()

        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels , out_channels=32 , kernel_size=3 , stride=1 , padding=1),
            nn.BatchNorm2d(132),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2 , stride=2)
          )

        # Second convolutional blcok
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32 , out_channels=64 , kernel_size=3 , stride=1 , padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2 , stride=2)
        )

        # Third convolutional block
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64 , out_channels=128 , kernel_size=3 , stride=1 , padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2 , stride=2)
        )

        # Flatten 3D feature map to 1D

        self.flatten = nn.Flatten()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=128 * 3 * 3 , out_features=256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=256 , out_features=128)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=128 , out_features=num_classes) # Output classes after classification

    def forward(self, x):
        # Pass input through convolutional layers
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Flatten the feature
        x = self.flatten(x)

        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

print(model)
