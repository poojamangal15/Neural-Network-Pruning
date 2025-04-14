import matplotlib.pyplot as plt

# Data from the user
steps = ['original', 'forward_3', 'backward_3', 'backward_2', 'backward_1']
accuracies = [0.915, 0.7602, 0.8071, 0.8231, 0.8303]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(steps, accuracies, marker='o', linestyle='-', linewidth=2, color='blue')
plt.title("Accuracy across Iterative Pruning and Reverse Rebuilding Steps")
plt.xlabel("Pruning/Rebuilding Step")
plt.ylabel("Accuracy")
plt.ylim(0.7, 0.95)
plt.grid(True)
plt.tight_layout()
plt.show()
