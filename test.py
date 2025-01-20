import matplotlib.pyplot as plt

# Data: After Pruning + Fine-Tuning
pruning_percentage = [20.0, 40.0, 60.0, 80.0]
acc_after_pruning = [0.7611, 0.7717, 0.7579, 0.7195]
f1_after_pruning  = [0.7590, 0.7674, 0.7565, 0.7143]

# Data: After Rebuilding + Fine-Tuning
acc_after_rebuild = [0.7828, 0.7761, 0.7660, 0.7682]
f1_after_rebuild  = [0.7811, 0.7764, 0.7647, 0.7687]

model_size_after_pruning = [48698799, 40518927, 32506240, 24667405]
model_size_after_rebuild = [57044810, 57044810, 57044810, 57044810]

plt.figure(figsize=(8, 5))

bar_width = 0.35
x = range(len(pruning_percentage))  # for the x-axis positions

# Bar for after pruning
plt.bar([p - bar_width/2 for p in x], model_size_after_pruning, 
        width=bar_width, color='orange', label='After Pruning + FT')

# Bar for after rebuilding
plt.bar([p + bar_width/2 for p in x], model_size_after_rebuild, 
        width=bar_width, color='cyan', label='After Rebuild + FT')

plt.xticks(x, [f"{int(p)}%" for p in pruning_percentage])  # labeling x-axis with the percentages
plt.title("Model Size vs. Pruning Percentage")
plt.ylabel("Number of Parameters")
plt.legend()
plt.grid(True, axis='y')
plt.show()
