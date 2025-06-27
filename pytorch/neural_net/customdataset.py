
import torch
from torch.utils.data import DataLoader, Dataset
# Question 13: Custom Dataset Class
class CustomDataset(Dataset):
    """
    Create a dataset class that can handle both regression and classification data.
    Include data augmentation options.

    Learning: Dataset creation, data preprocessing pipeline
    """
    def __init__(self, X, y, transform=None, task_type='classification'):
        self.X = X
        self.y = y
        self.transform = transform
        self.task_type = task_type

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx] , self.y[idx]
        if self.transform:
          sample = self.transform(sample)
        if self.task_type == 'regression':
          return sample[0] , sample[1].to(torch.float32)
        if self.task_type == 'classification':
          return sample[0] , sample[1].to(torch.long)

torch.manual_seed(42)
X = torch.randn(4 , 8)
y = torch.tensor([0, 1, 2, 0])

print(f"\n{'='*60}")
print("Raw Data")
print(f"The input data is {X}")
print(f"The output data is {y}")
print(f"{'='*60}")




dataset1 = CustomDataset(X , y , False , 'regression')
dataloader1 = DataLoader(dataset1)
total_samples = len(dataset1)

print(f"\n{'='*60}")
print("Data for Regression Task")
for i , (targets , labels) in enumerate(dataloader1):
  print(f"The input data is {targets}")
  print(f"The output data is {labels}")
print(f"{'='*60}")


dataset2 = CustomDataset(X , y , False , 'classification')
dataloader2 = DataLoader(dataset2)
total_samples = len(dataset2)

print(f"\n{'='*60}")
print("Data for Classification Task")

for i , (targets , labels) in enumerate(dataloader2):
  print(f"The input data is {targets}")
  print(f"The output data is {labels}")
print(f"{'='*60}")
