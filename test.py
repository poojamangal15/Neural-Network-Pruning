import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

# Data from the output
iterations = [31250000, 62500000, 125000000, 250000000, 500000000, 1000000000, 2000000000]
execution_times = [0.038078, 0.079033, 0.152449, 0.279447, 0.561927, 1.103204, 2.241537]

# Create the plot with annotations in millions
plt.figure(figsize=(10, 6))
plt.plot(iterations, execution_times, marker='o', linestyle='-', label='Execution Time')

# Adding annotations for each point in millions
for x, y in zip(iterations, execution_times):
    plt.text(x, y, f'{x//1_000_000}M', fontsize=9, ha='right', va='bottom')

# Adding labels and title
plt.xlabel('Number of Iterations (niter)', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.title('Execution Time vs. Number of Iterations', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)

# Show the plot
plt.show()

