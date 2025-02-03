import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame from your results (example data, update with your real values)
data = {
    "Method": ["Basic Dep FT", "Basic Dep FT+Rebuild", "Layer-wise Mild FT", "Layer-wise Mild FT+Rebuild", 
               "Layer-wise Agg FT", "Layer-wise Agg FT+Rebuild", 
               "Iterative Forward", "Iterative Backward 1", "Iterative Backward 2", "Iterative Backward 3"],
    "Test Accuracy": [0.8958, 0.7736, 0.8913, 0.7795, 0.8708, 0.758, 0.7551, 0.7637, 0.7462, 0.7488],
    "Model Size (MB)": [185.78, 217.61, 156.65, 217.61, 183.64, 217.61, 92.95, 135.93, 185.78, 217.61]
}

df = pd.DataFrame(data)

# # Bar plot for Test Accuracy
# plt.figure(figsize=(10,6))
# plt.bar(df["Method"], df["Test Accuracy"], color="skyblue")
# plt.ylabel("Test Accuracy")
# plt.title("Test Accuracy Comparison Across Pruning Methods")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()

# # Bar plot for Model Size
# plt.figure(figsize=(10,6))
# plt.bar(df["Method"], df["Model Size (MB)"], color="salmon")
# plt.ylabel("Model Size (MB)")
# plt.title("Model Size Comparison Across Pruning Methods")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()


import seaborn as sns

sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="Model Size (MB)", y="Test Accuracy", hue="Method", s=100)
plt.title("Accuracy vs. Model Size Across Pruning Methods")
plt.tight_layout()
plt.show()
