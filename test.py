import matplotlib.pyplot as plt
# Define the filters for both models including linear layers
original_model_filters = [64, 192, 384, 256, 256, 9216]
dense_model_filters = [45, 134, 269, 179, 179, 6444]

# Layer names including linear layers
layers_with_linear = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Linear 1']

# Create the bar chart including linear layers
plt.figure(figsize=(12, 6))
x = range(len(layers_with_linear))
width = 0.35

plt.bar([p - width / 2 for p in x], original_model_filters, width, label='Original Model')
plt.bar([p + width / 2 for p in x], dense_model_filters, width, label='Dense Model')

plt.xticks(x, layers_with_linear, rotation=45, ha='right')
plt.xlabel('Layers')
plt.ylabel('Number of Filters/Neurons')
plt.title('Comparison of Filters and Linear Layers: Original vs Dense Model')
plt.legend()
plt.tight_layout()

# Save the chart
# file_path_with_linear = '/mnt/data/filters_comparison_with_linear.png'
# plt.savefig(file_path_with_linear)
plt.show()

# file_path_with_linear
