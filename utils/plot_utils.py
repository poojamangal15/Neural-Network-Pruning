import matplotlib.pyplot as plt

def plot_metrics(metrics):
    print("Metrics Debug:")
    print("Pruning Percentages:", metrics.get("pruning_percentage", []))
    print("Test Accuracy:", metrics.get("test_accuracy", []))
    print("F1 Score:", metrics.get("f1_score", []))
    print("Model Size:", metrics.get("model_size", []))

    if "pruning_percentage" in metrics and "test_accuracy" in metrics:
        plt.figure()
        plt.plot(metrics["pruning_percentage"], metrics["test_accuracy"], marker='o', label="Accuracy")
        plt.title("Test Accuracy vs. Pruning Percentage")
        plt.xlabel("Pruning Percentage (%)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.show()
