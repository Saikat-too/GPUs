
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# Question 19: Attention Mechanism
class SimpleAttention(nn.Module):
    """
    Implement a basic attention mechanism.

    Learning: Attention computation, weighted aggregation
    """
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # Linear transformation for query , key and value
        self.query = nn.Linear(hidden_size , hidden_size)
        self.key   = nn.Linear(hidden_size , hidden_size)
        self.value = nn.Linear(hidden_size , hidden_size)

        # Ouput
        self.output = nn.Linear(hidden_size , hidden_size)

        # Scaling factor for dot product attention
        self.scale = math.sqrt(hidden_size)

    def forward(self, query, key, value , mask=None):
      """
        Forward pass of the attention mechanism.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, hidden_size).
            key (torch.Tensor): Key tensor of shape (batch_size, hidden_size).
            value (torch.Tensor): Value tensor of shape (batch_size, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len_q , hidden_size).
            attention weights : torch.Tensor: Attention weights of shape (batch_size, seq_len_q, seq_len_k).
      """
      # Step 1 : Linear Transoformation
      query = self.query(query)
      key  = self.key(key)
      value = self.value(value)

      # Step 2 : Compute attention score
      scores = torch.bmm(query , key.transpose(1,2))

      # Step 3 : Scale the scores
      scores_scalled = scores / self.scale

      # Step 4 : Softmax to get weights
      A = F.softmax(scores_scalled , dim=1)

      # Step 5 : Apply attention weights to values
      context = torch.bmm(A , value)

      # Step 6 : Final output
      output = self.output(context)


      return output , A





# Parameters
batch_size = 2
seq_len = 5
hidden_size = 64

# Create model
attention = SimpleAttention(hidden_size)

# Create sample data
query = torch.randn(batch_size, seq_len, hidden_size)
key = torch.randn(batch_size, seq_len, hidden_size)
value = torch.randn(batch_size, seq_len, hidden_size)

# Forward pass
output, attention_weights = attention(query, key, value)

print(f"Input shapes:")
print(f"Query: {query.shape}")
print(f"Key: {key.shape}")
print(f"Value: {value.shape}")
print()
print(f"Output shapes:")
print(f"Output: {output.shape}")
print(f"Attention weights: {attention_weights.shape}")
print()

# Verify attention weights sum to 1
weights_sum = attention_weights.sum(dim=-1)
print(f"Attention weights sum (should be ~1.0): {weights_sum[0, 0]:.4f}")
