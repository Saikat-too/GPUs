
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
# Question 16: Weight Initialization
def initialize_weights(model, strategy='xavier_uniform', activation='relu'):
    """
    Implement different weight initialization strategies:
    - Xavier/Glorot initialization
    - He initialization
    - Custom initialization based on layer type

    Learning: Initialization importance, different strategies
    """

    for module in model.modules():
        if isinstance(module, nn.Linear):
            if strategy == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain(activation))
            elif strategy == 'xavier_normal':
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain(activation))
            elif strategy == 'he_uniform':
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity=activation)
            elif strategy == 'he_normal':
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity=activation)
            elif strategy == 'custom':
                fan_in = module.in_features
                fan_out = module.out_features

                if fan_in > 1000:
                    std = np.sqrt(1.0 / (fan_in + fan_out))
                else:
                    std = np.sqrt(2.0 / (fan_in + fan_out))

                nn.init.normal_(module.weight, mean=0.0, std=std)
            else:
                raise ValueError("Invalid initialization strategy")

            # Check if bias exists before initializing
            if module.bias is not None:
                module.bias.data.fill_(0)

models = {}
strategies = ['xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal', 'custom']

for strategy in strategies:
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    initialize_weights(model, strategy=strategy, activation='relu')
    models[strategy] = model

# Analyze weight distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (strategy, model) in enumerate(models.items()):
    weights = []
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            weights.extend(layer.weight.data.flatten().numpy())

    axes[i].hist(weights, bins=50, alpha=0.7, density=True)
    axes[i].set_title(f'{strategy}\nMean: {np.mean(weights):.4f}, Std: {np.std(weights):.4f}')
    axes[i].set_xlabel('Weight Value')
    axes[i].set_ylabel('Density')
    axes[i].grid(True, alpha=0.3)

# Remove empty subplot
fig.delaxes(axes[5])
plt.tight_layout()
plt.show()
