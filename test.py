# Re-import necessary libraries after code state reset
import matplotlib.pyplot as plt
import numpy as np

# Pruning percentages
pruning_percents = [30, 50, 70]

# Model sizes (MB) for VGG
vgg_original = [58.24] * 3
vgg_pruned = [29.45, 15.66, 6.49]
vgg_rebuilt = [58.24] * 3

# Model sizes (MB) for AlexNet
alexnet_original = [217.61] * 3
alexnet_pruned = [170.38, 138.91, 108.69]
alexnet_rebuilt = [217.61] * 3

# Model sizes (MB) for ResNet-20
resnet_original = [1.08] * 3
resnet_pruned = [0.531, 0.282, 0.128]
resnet_rebuilt = [1.08] * 3

# Plotting
bar_width = 0.25
x = np.arange(len(pruning_percents))

fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

# VGG Plot
axs[0].bar(x - bar_width, vgg_original, width=bar_width, label="Original", color='gray')
axs[0].bar(x, vgg_pruned, width=bar_width, label="Pruned + FT", color='orange')
axs[0].bar(x + bar_width, vgg_rebuilt, width=bar_width, label="Rebuilt + FT", color='cyan')
axs[0].set_title("VGG-16")
axs[0].set_xticks(x)
axs[0].set_xticklabels([f"{p}%" for p in pruning_percents])
axs[0].set_ylabel("Model Size (MB)")
axs[0].grid(True, axis='y', linestyle='--', alpha=0.7)

# AlexNet Plot
axs[1].bar(x - bar_width, alexnet_original, width=bar_width, color='gray')
axs[1].bar(x, alexnet_pruned, width=bar_width, color='orange')
axs[1].bar(x + bar_width, alexnet_rebuilt, width=bar_width, color='cyan')
axs[1].set_title("AlexNet")
axs[1].set_xticks(x)
axs[1].set_xticklabels([f"{p}%" for p in pruning_percents])
axs[1].grid(True, axis='y', linestyle='--', alpha=0.7)

# ResNet Plot (rescaled for better visibility)
axs[2].bar(x - bar_width, resnet_original, width=bar_width, color='gray')
axs[2].bar(x, resnet_pruned, width=bar_width, color='orange')
axs[2].bar(x + bar_width, resnet_rebuilt, width=bar_width, color='cyan')
axs[2].set_title("ResNet-20")
axs[2].set_xticks(x)
axs[2].set_xticklabels([f"{p}%" for p in pruning_percents])
axs[2].grid(True, axis='y', linestyle='--', alpha=0.7)

# Common legend and layout
fig.suptitle("Comparing Model Size Across Pruning Stages")
fig.legend(["Original", "Pruned", "Rebuilt"], loc="lower center", ncol=3, fontsize=12)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])

plt.show()
