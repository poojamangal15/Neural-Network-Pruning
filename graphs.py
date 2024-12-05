import matplotlib.pyplot as plt

# Pruning percentages
pruning_percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Accuracies for different methods
accuracy_global_unstructured = [0.8891000151634216, 0.8895999789237976, 0.8899999856948853, 0.890500009059906, 0.8881999850273132, 0.8899999856948853]
accuracy_depGraph = [0.8889776357827476, 0.8921725239616614, 0.8927715654952076, 0.8900758785942492, 0.8763977635782748, 0.8740015974440895]
accuracy_random_unstructured = [0.8891000151634216, 0.6327999830245972, 0.38530001044273376, 0.18219999969005585, 0.16740000247955322, 0.10450000315904617]
accuracy_structured = [0.8891000151634216, 0.7013000249862671, 0.22390000522136688, 0.1559000015258789, 0.11100000143051147, 0.09769999980926514]
accuracy_unstructured = [0.8891000151634216, 0.8901000022888184, 0.8884999752044678, 0.8847000002861023, 0.8628000020980835, 0.7562000155448914]
high_level_pruner = [0.8891000151634216, 0.7578125, 0.2890625, 0.1796875, 0.171875, 0.078125]
# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(pruning_percentages, accuracy_global_unstructured, marker='o', label='Global Unstructured')
plt.plot(pruning_percentages, accuracy_depGraph, marker='o', label='Dependency Graph')
plt.plot(pruning_percentages, accuracy_random_unstructured, marker='o', label='Random Unstructured')
plt.plot(pruning_percentages, accuracy_structured, marker='o', label='Structured')
plt.plot(pruning_percentages, accuracy_unstructured, marker='o', label='Unstructured')
plt.plot(pruning_percentages, high_level_pruner, marker='o', label='High Level Pruner')

# Adding labels, title, legend, and grid
plt.xlabel('Pruning Percentage')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Pruning Percentage for Different Methods')
plt.legend()
plt.grid(True)

# Display the graph
plt.show()
