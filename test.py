import matplotlib.pyplot as plt

# Data from the iterative pruning results
steps = ["original", "forward_3", "backward_3", "backward_2", "backward_1"]
accuracy = [0.9016, 0.7755, 0.7551, 0.7623, 0.7839]
params = [272474, 72932, 110840, 174504, 272474]
size_mb = [1.0786, 0.3151, 0.4624, 0.7065, 1.0786]

# Create the plots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Accuracy Progression Over Steps
axs[0].plot(steps, accuracy, marker='o', color='blue', linestyle='-', label='Accuracy')
axs[0].set_ylabel("Accuracy")
axs[0].set_title("Accuracy Progression Over Pruning Steps")
axs[0].grid(True)
axs[0].legend()

# Parameter Count Over Steps
axs[1].plot(steps, params, marker='o', color='red', linestyle='-', label='Parameters')
axs[1].set_ylabel("Number of Parameters")
axs[1].set_title("Parameter Count Over Pruning Steps")
axs[1].grid(True)
axs[1].legend()

# Model Size Over Steps
axs[2].plot(steps, size_mb, marker='o', color='green', linestyle='-', label='Model Size (MB)')
axs[2].set_xlabel("Pruning Steps")
axs[2].set_ylabel("Model Size (MB)")
axs[2].set_title("Model Size Over Pruning Steps")
axs[2].grid(True)
axs[2].legend()

# Adjust layout and show plots
plt.tight_layout()
plt.show()
