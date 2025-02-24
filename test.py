import matplotlib.pyplot as plt

# Data
stage_labels = ["Initial", "Pruned", "Rebuilt"]
accuracies = [0.9016, 0.7876, 0.8246]

# Create the figure and axis
plt.figure(figsize=(6, 4))
plt.plot(stage_labels, accuracies, marker='o', linestyle='-', color='blue', label='Accuracy')

# (Alternative) If you’d prefer a bar chart, you could do:
# plt.bar(stage_labels, accuracies, color=['green', 'red', 'blue'])

# Add chart details
plt.title("Model Accuracy at Each Stage")
plt.xlabel("Model Stage")
plt.ylabel("Accuracy")
plt.ylim(0.0, 1.0)  # So that we can clearly see the differences
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()
