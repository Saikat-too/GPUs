
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , TensorDataset

# Question 14: Training Loop Implementation
class Trainer:
    """
    Implement a flexible training loop with:
    - Training and validation phases
    - Metrics tracking
    - Early stopping
    - Learning rate scheduling

    Learning: Training best practices, monitoring, optimization
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for data , targets in self.train_loader:
          data , targets = data.to(self.device) , targets.to(self.device).float()

        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs.squeeze() , targets)
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        epoch_loss = running_loss / num_batches
        self.train_losses.append(epoch_loss)
        return epoch_loss

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        with torch.no_grad():
          for data , targets in self.val_loader:
            data , targets = data.to(self.device) , targets.to(self.device).float()

            outputs = self.model(data)
            loss = self.criterion(outputs.squeeze() , targets)

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / num_batches
        self.val_losses.append(epoch_loss)
        return epoch_loss


    def train(self, num_epochs, early_stopping_patience=None):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
          train_loss = self.train_epoch()
          val_loss   = self.validate_epoch()

          print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

          if early_stopping_patience is not None:
            if val_loss < best_val_loss:
              best_val_loss = val_loss
              patience_counter = 0

            else:
              patience_counter += 1
              if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break



        print("Training Completed")




X_train = torch.randn(1000, 10)
y_train = torch.randint(0, 2, (1000,))
X_val = torch.randn(200, 10)
y_val = torch.randint(0, 2, (200,))

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)


trainer.train(50 , 10)
