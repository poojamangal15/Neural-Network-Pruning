import torch
import torch.nn as nn
from copy import deepcopy
from models.depGraph_fineTuner import DepGraphFineTuner


def count_parameters(model):
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_model_during_training(model, input_size):
    flops, macs, params = get_model_profile(model=model, input_shape=input_size, print_profile=True, as_string=False)
    size_mb = model_size_in_mb(model)
    print(f"Params: {params / 1e6:.3f} M, MACs: {macs / 1e9:.3f} GMACs, FLOPs: {flops / 1e9:.3f} GFLOPs, Size: {size_mb:.2f} MB")
    return params, macs, size_mb

def model_size_in_mb(model):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / (1024 * 1024)
    os.remove("temp.p")
    return size_mb

def get_pruned_info(groups, original_model):
    """
    Collect pruned indices and total counts for each layer in the model.

    Parameters:
    - groups: List of pruning groups obtained during pruning.

    Returns:
    - pruned_info: A dictionary containing pruned indices and counts for each layer.
    - num_pruned_channels: A dictionary containing the total number of pruned channels (dim0 and dim1) for each layer.
    - pruned_weights: A dictionary containing the pruned weights for each layer.

    """
    pruned_info = {}
    num_pruned_channels = {}
    pruned_weights = {}

    module_dict = dict(original_model.named_modules())

    for layer_name, group in groups:
        
        if layer_name.startswith("model."):
            layer_name = layer_name[len("model."):]
        # Initialize the layer in the dictionary
        if layer_name not in pruned_info:
            pruned_info[layer_name] = {'pruned_dim0': [], 'pruned_dim1': []}

        for item in group.items:
            # Extract pruned indices
            pruned_indices = item.idxs  # Indices of pruned elements

            # Check if this pruning affects dim0 (output channels) or dim1 (input channels)
            if "out_channels" in str(item):  # Output channels pruning
                pruned_info[layer_name]['pruned_dim0'].extend(pruned_indices)

                if "in_channels" in str(item):
                    pruned_info[layer_name]['pruned_dim1'].extend(pruned_indices)
            elif "in_channels" in str(item):  # Input channels pruning
                pruned_info[layer_name]['pruned_dim1'].extend(pruned_indices)

        pruned_dim0 = pruned_info[layer_name]['pruned_dim0']
        pruned_dim1 = pruned_info[layer_name]['pruned_dim1']
        pruned_dim0_count = len(pruned_dim0)
        pruned_dim1_count = len(pruned_dim1)
        num_pruned_channels[layer_name] = (pruned_dim0_count, pruned_dim1_count)

        layer = module_dict.get(layer_name)
        if layer is None:
            print(f"Layer {layer_name} not found in model. Skipping.")
            pruned_weights[layer_name] = torch.empty((0, 0))
            continue

        if isinstance(layer, torch.nn.Conv2d):
            weights = layer.weight.detach()
            if pruned_dim0 or pruned_dim1:  # If there are pruned dimensions
                pruned_weight_tensor = weights[pruned_dim0][:, pruned_dim1, :, :]
                pruned_weights[layer_name] = pruned_weight_tensor
        elif isinstance(layer, torch.nn.Linear):
            weights = layer.weight.detach()
            if pruned_dim0 or pruned_dim1:  # If there are pruned dimensions
                pruned_weight_tensor = weights[pruned_dim0][:, pruned_dim1]
                pruned_weights[layer_name] = pruned_weight_tensor
        
            # for name, layer in original_model.named_modules():
    #     # Access the original unpruned weights
    #     print("LAYER-----", layer)
    #     weights = layer.weight.detach()
    #     print(f"Layer: {layer_name}, Weights shape: {weights.shape}")

    #     pruned_dim0 = pruned_info[name]['pruned_dim0']
    #     pruned_dim1 = pruned_info[name]['pruned_dim1']
    #     if pruned_dim0 or pruned_dim1:
    #         try:
    #             pruned_weights[name] = weights[pruned_dim0][:, pruned_dim1, :, :]
    #         except IndexError as e:
    #             print(f"IndexError for layer {name}: {e}. Skipping weights.")
    #             pruned_weights[name] = None
    #     else:
    #         pruned_weights[name] = None

    for layer_name in pruned_info:
        pruned_info[layer_name]['pruned_dim0'] = sorted(pruned_info[layer_name]['pruned_dim0'])
        pruned_info[layer_name]['pruned_dim1'] = sorted(pruned_info[layer_name]['pruned_dim1'])


    return pruned_info, num_pruned_channels, pruned_weights


def get_unpruned_info(groups, original_model):
    unpruned_info = {}
    num_unpruned_channels = {}
    unpruned_weights = {}

    # Access the actual model inside DepGraphFineTuner
    module_dict = dict(original_model.named_modules())

    for layer_name, group in groups:
        # Remove the "model." prefix if it exists
        if layer_name.startswith("model."):
            layer_name = layer_name[len("model."):]

        if layer_name not in unpruned_info:
            unpruned_info[layer_name] = {'unpruned_dim0': [], 'unpruned_dim1': []}

        # Total indices (output and input channels)
        total_output_channels = None
        total_input_channels = None

        layer = module_dict.get(layer_name)
        if isinstance(layer, torch.nn.Conv2d):
            total_output_channels = layer.weight.shape[0]
            total_input_channels = layer.weight.shape[1]
        elif isinstance(layer, torch.nn.Linear):
            total_output_channels = layer.weight.shape[0]
            total_input_channels = layer.weight.shape[1]

        # Get all indices
        all_output_indices = list(range(total_output_channels)) if total_output_channels else []
        all_input_indices = list(range(total_input_channels)) if total_input_channels else []

        # Determine pruned indices from the group
        pruned_dim0 = []
        pruned_dim1 = []
        for item in group.items:
            pruned_indices = item.idxs
            if "out_channels" in str(item):
                pruned_dim0.extend(pruned_indices)
            elif "in_channels" in str(item):
                pruned_dim1.extend(pruned_indices)

        # Determine unpruned indices by subtracting pruned indices
        unpruned_dim0 = [i for i in all_output_indices if i not in pruned_dim0]
        unpruned_dim1 = [i for i in all_input_indices if i not in pruned_dim1]

        # Populate unpruned info
        unpruned_info[layer_name]['unpruned_dim0'] = unpruned_dim0
        unpruned_info[layer_name]['unpruned_dim1'] = unpruned_dim1
        num_unpruned_channels[layer_name] = (len(unpruned_dim0), len(unpruned_dim1))

        # Get weights of unpruned neurons
        if layer is not None:
            weights = layer.weight.detach().cpu()
            if isinstance(layer, torch.nn.Conv2d):
                unpruned_weights[layer_name] = weights[unpruned_dim0][:, unpruned_dim1, :, :]
            elif isinstance(layer, torch.nn.Linear):
                unpruned_weights[layer_name] = weights[unpruned_dim0][:, unpruned_dim1]
        else:
            print(f"Layer {layer_name} not found. Skipping.")
            unpruned_weights[layer_name] = None

    return unpruned_info, num_unpruned_channels, unpruned_weights
