import matplotlib.pyplot as plt

# Data
pruning_percentage = [20.0, 40.0, 60.0, 80.0]
model_size_original = [57044810] * len(pruning_percentage)  # Original size repeated for all pruning percentages
model_size_after_pruning = [48698799, 40518927, 32506240, 24667405]
model_size_after_rebuild = [57044810, 57044810, 57044810, 57044810]

# Bar plot setup
bar_width = 0.25
x = range(len(pruning_percentage))  # X-axis positions for the bars

plt.figure(figsize=(10, 6))

# Bars for original model size
plt.bar([p - bar_width for p in x], model_size_original, 
        width=bar_width, color='gray', label='Original Number of Parameters')

# Bars for after pruning
plt.bar(x, model_size_after_pruning, 
        width=bar_width, color='orange', label='After Pruning + FT')

# Bars for after rebuilding
plt.bar([p + bar_width for p in x], model_size_after_rebuild, 
        width=bar_width, color='cyan', label='After Rebuild + FT')

# X-axis labels and title
plt.xticks(x, [f"{int(p)}%" for p in pruning_percentage])  # Labeling with pruning percentages
plt.title("Number of Parameters vs. Pruning Percentage")
plt.xlabel("Pruning Percentage (%)")
plt.ylabel("Number of Parameters")
plt.legend()
plt.grid(True, axis='y')

plt.show()
