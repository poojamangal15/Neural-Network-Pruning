# import matplotlib.pyplot as plt

# # Updated iterative accuracy data
# iterative_data = {
#     "ResNet-56": {
#         "steps": ['original', 'forward_3', 'backward_3', 'backward_2', 'backward_1'],
#         "accuracy": [0.8649681528662421, 0.7179617834394905, 0.7577070063694268, 0.7864968152866242, 0.8012738853503185]
#     },
#     "ResNet-20": {
#         "steps": ['original', 'forward_3', 'backward_3', 'backward_2', 'backward_1'],
#         "accuracy": [0.915, 0.7602, 0.8071, 0.8231, 0.8303]
#     },
#     "AlexNet": {
#         "steps": ['original', 'forward_3', 'backward_3', 'backward_2', 'backward_1'],
#         "accuracy": [0.9016, 0.7648, 0.8286, 0.8567, 0.8625]
#     },
#     "VGG-16": {
#         "steps": ['original', 'forward_3', 'backward_3', 'backward_2', 'backward_1'],
#         "accuracy": [0.9394, 0.8587, 0.8599, 0.8679, 0.8802]
#     }
# }

# # Plotting
# fig, axes = plt.subplots(2, 2, figsize=(9, 7))
# fig.suptitle("Iterative Pruning and Rebuilding Accuracy", fontsize=18)

# # Map models to subplot positions
# positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

# for (model, data), pos in zip(iterative_data.items(), positions):
#     ax = axes[pos]
#     ax.plot(data["steps"], data["accuracy"], marker='o', linestyle='-', linewidth=2)
#     ax.set_title(model, fontsize=14)
#     ax.set_xlabel("Step", fontsize=12)
#     ax.set_ylabel("Accuracy", fontsize=12)
#     ax.set_xticks(range(len(data["steps"])))
#     ax.set_xticklabels(data["steps"], rotation=30)
#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.set_ylim(min(data["accuracy"]) - 0.05, 1.0)

# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Model data for all three networks
data = {
    "VGG-16": {
        "pruning": ["30%", "50%", "70%"],
        "original": {"size": [58.24]*3, "params": [15253578]*3},
        "pruned": {"size": [28.50, 14.61, 5.24], "params": [7458867, 3818538, 1363717]},
        "rebuilt": {"size": [58.24]*3, "params": [15253578]*3}
    },
    "AlexNet": {
        "pruning": ["30%", "50%", "70%"],
        "original": {"size": [217.61]*3, "params": [57044810]*3},
        "pruned": {"size": [119.20, 35.99, 5.39], "params": [31245867, 9434983, 1410371]},
        "rebuilt": {"size": [217.61]*3, "params": [57044810]*3}
    },
    "ResNet-20": {
        "pruning": ["30%", "50%", "70%"],
        "original": {"size": [1.08]*3, "params": [272474]*3},
        "pruned": {"size": [0.53, 0.30, 0.13], "params": [129359, 68786, 23580]},
        "rebuilt": {"size": [1.08]*3, "params": [272474]*3}
    }
}

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
fig.suptitle("Model Size and Parameter Count across Pruning Levels", fontsize=16)

bar_width = 0.25
x = np.arange(3)

for i, (model, vals) in enumerate(data.items()):
    # Model size
    axes[0, i].bar(x - bar_width, vals["original"]["size"], width=bar_width, label='Original', color='gray')
    axes[0, i].bar(x, vals["pruned"]["size"], width=bar_width, label='Pruned', color='orange')
    axes[0, i].bar(x + bar_width, vals["rebuilt"]["size"], width=bar_width, label='Rebuilt', color='cyan')
    axes[0, i].set_title(model)
    axes[0, i].set_xticks(x)
    axes[0, i].set_xticklabels(vals["pruning"])
    axes[0, i].set_ylabel("Model Size (MB)")
    # axes[0, i].grid(True, linestyle='--', alpha=0.6)

    # Parameter count (converted to millions)
    original_params = np.array(vals["original"]["params"]) / 1e6
    pruned_params = np.array(vals["pruned"]["params"]) / 1e6
    rebuilt_params = np.array(vals["rebuilt"]["params"]) / 1e6

    axes[1, i].bar(x - bar_width, original_params, width=bar_width, color='gray')
    axes[1, i].bar(x, pruned_params, width=bar_width, color='orange')
    axes[1, i].bar(x + bar_width, rebuilt_params, width=bar_width, color='cyan')
    axes[1, i].set_title(model)
    axes[1, i].set_xticks(x)
    axes[1, i].set_xticklabels(vals["pruning"])
    axes[1, i].set_ylabel("Params (Millions)")
    # axes[1, i].grid(True, linestyle='--', alpha=0.6)

# Common legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()
