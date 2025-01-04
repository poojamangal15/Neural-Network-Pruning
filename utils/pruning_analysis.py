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
            elif "in_channels" in str(item):  # Input channels pruning
                pruned_info[layer_name]['pruned_dim1'].extend(pruned_indices)

        num_pruned_channels[layer_name] = (len(pruned_info[layer_name]['pruned_dim0']), len(pruned_info[layer_name]['pruned_dim1']))

        print(f"Layer name from groups: {layer_name}")

        layer = module_dict.get(layer_name)
        weights = layer.weight.detach()
        print(f"Layer {layer_name} weights have shape: {weights.shape}.")

        if weights.dim() != 4:  # Ensure weights have the expected shape
            print(f"Layer {layer_name} weights have unexpected shape: {weights.shape}. Skipping.")
            pruned_weights[layer_name] = None
            continue

        pruned_dim0 = pruned_info[layer_name]['pruned_dim0']
        pruned_dim1 = pruned_info[layer_name]['pruned_dim1']
        if pruned_dim0 or pruned_dim1:
            try:
                pruned_weights[layer_name] = weights[pruned_dim0][:, pruned_dim1, :, :]
            except IndexError as e:
                print(f"IndexError for layer {layer_name}: {e}. Skipping weights.")
                pruned_weights[layer_name] = None
        else:
            pruned_weights[layer_name] = None
        
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


def get_dependency_pruned_info(model, layers_to_prune, pruned_info):
    """
    Extracts and organizes pruning information for dependency-based pruning to align with your supervisor's code.

    Args:
        model (torch.nn.Module): The pruned model.
        layers_to_prune (dict): Dictionary of layers that were subject to pruning.
        pruned_info (dict): Dictionary tracking pruned layers and indices.

    Returns:
        tuple: A tuple containing:
            - pruned_info (dict): Updated dictionary with layer names, pruned indices for dim=0 and dim=1.
            - num_pruned_channels (dict): Dictionary with the count of pruned channels for dim=0 and dim=1 for each layer.
            - pruned_weights (dict): Dictionary containing tensors of pruned weights for each layer.
    """
    num_pruned_channels = {}
    pruned_weights = {}

    for layer_name, layer_module in layers_to_prune.items():
        layer_info = {}

        # Get the weight tensor
        weights = layer_module.weight.detach()

        # Identify pruned filters along dim=0 (output channels)
        pruned_dim0 = torch.nonzero(weights.abs().sum(dim=(1, 2, 3)) == 0).squeeze().tolist()
        if isinstance(pruned_dim0, int):  # Handle single index case
            pruned_dim0 = [pruned_dim0]
        layer_info['pruned_dim0'] = pruned_dim0

        # Identify pruned filters along dim=1 (input channels)
        pruned_dim1 = torch.nonzero(weights.abs().sum(dim=(0, 2, 3)) == 0).squeeze().tolist()
        if isinstance(pruned_dim1, int):  # Handle single index case
            pruned_dim1 = [pruned_dim1]
        layer_info['pruned_dim1'] = pruned_dim1

        # Update the pruned_info dictionary
        print(f"Type of pruned_info: {type(pruned_info)}")

        print("Pruning info 2", pruned_info)

        pruned_info[layer_name] = layer_info

        # Count pruned channels
        num_pruned_channels[layer_name] = (len(pruned_dim0), len(pruned_dim1))

        # Extract pruned weights as a tensor
        pruned_layer_weights = weights[pruned_dim0][:, pruned_dim1, :, :]
        pruned_weights[layer_name] = pruned_layer_weights

    return pruned_info, num_pruned_channels, pruned_weights


import copy

def normalize_layer_name(layer_name):
    # If the pruned info stores 'model.features.3', but named_modules is 'features.3', remove 'model.'
    if layer_name.startswith("model."):
        return layer_name[len("model."):]
    return layer_name

def rebuild_model(pruned_model, pruned_info):
    """
    Rebuild a bigger model from a pruned_model + info about pruned channels.
    """
    # 1) Create a fresh copy of the original architecture with the full channel count
    # For example, if your original model was "AlexNet()", do:
    checkpoint_path = "./checkpoints/best_checkpoint.ckpt"
    # Load DepGraphFineTuner instead of AlexNetFineTuner so we can call fine_tune_model later
    rebuilt_model = DepGraphFineTuner.load_from_checkpoint(checkpoint_path)
    rebuilt_model.eval()

    # 2) Get submodules
    pruned_named  = dict(pruned_model.named_modules())   # e.g. "model.features.3"
    rebuilt_named = dict(rebuilt_model.named_modules())  # same keys in the full model

    # 3) Re-insert for each layer entry in pruned_info
    for layer_key, info in pruned_info.items():
        # e.g. layer_key = "model.features.3_out" or "model.features.4_in"
        # or maybe just "model.features.3" with info["dim"] = 0 or 1
        dim = info["dim"]  # 0 => out-ch, 1 => in-ch
        pruning_idxs = info["pruning_idxs"]   # the pruned channel indices (Python list)
        pruned_w_data = info["weights"]       # pruned weight data (torch.Tensor)
        pruned_b_data = info.get("bias", None)

        # Figure out the submodule name (strip any "_out" or "_in" suffix)
        # e.g. "model.features.3_out" -> "model.features.3"
        # or if you just store "model.features.3" with "dim=0", skip this step
        base_name = layer_key
        if base_name.endswith("_out"):
            base_name = base_name.replace("_out", "")
        elif base_name.endswith("_in"):
            base_name = base_name.replace("_in", "")

        if base_name not in pruned_named or base_name not in rebuilt_named:
            print(f"Layer {base_name} not found in pruned/rebuilt model, skipping.")
            continue

        pruned_layer  = pruned_named[base_name]
        rebuilt_layer = rebuilt_named[base_name]
        # pruned_layer.weight has shape after pruning
        # rebuilt_layer.weight is the original bigger shape

        # 4) Indices of channels that remain
        out_dim_size = rebuilt_layer.weight.shape[dim]  # the bigger dimension count
        all_indices  = set(range(out_dim_size))
        pruned_set   = set(pruning_idxs)
        remain_set   = all_indices - pruned_set
        remain_indices = sorted(list(remain_set))

        # For convenience, turn the pruned & remain lists into arrays
        import torch
        pruned_idxs_torch = torch.tensor(pruning_idxs, dtype=torch.long)
        remain_idxs_torch = torch.tensor(remain_indices, dtype=torch.long)

        # 5) We'll build new_weight that is the same shape as the rebuilt_layer
        new_weight = rebuilt_layer.weight.data.clone()  # shape => original size
        if rebuilt_layer.bias is not None:
            new_bias = rebuilt_layer.bias.data.clone()
        else:
            new_bias = None

        # pruned_layer weight => shape matches pruned dimension
        # e.g. for dim=0, pruned_layer.weight.shape[0] == len(remain_indices)
        # e.g. for dim=1, pruned_layer.weight.shape[1] == len(remain_indices)
        old_w_data = pruned_layer.weight.data

        # 5a) Copy remain channels from pruned_layer
        # We only copy from pruned_layer because those "remain" channels were kept after pruning & fine-tuning
        # For dim=0 => out-ch, use advanced indexing
        # For dim=1 => in-ch, similarly
        if dim == 0:
            # out-ch remain
            new_weight[remain_idxs_torch] = old_w_data
            # Also copy remain bias if exists
            if new_bias is not None and pruned_layer.bias is not None:
                new_bias[remain_idxs_torch] = pruned_layer.bias.data
        elif dim == 1:
            # in-ch remain => we do new_weight[:, remain_idxs, ...] = old_w_data
            new_weight[:, remain_idxs_torch] = old_w_data
            # There's no "bias" along in-ch, so typically no bias update for dim=1
        else:
            raise ValueError("dim must be 0 or 1 for Conv2d.")

        # 5b) Re-insert pruned channels from pruned_w_data
        # pruned_w_data shape => #pruned channels in that dimension
        if dim == 0:
            # out-ch re-insert
            for i, p_idx in enumerate(pruning_idxs):
                new_weight[p_idx] = pruned_w_data[i]
            # Also re-insert bias
            if new_bias is not None and pruned_b_data is not None:
                for i, p_idx in enumerate(pruning_idxs):
                    new_bias[p_idx] = pruned_b_data[i]
        else:
            # in-ch re-insert
            for i, p_idx in enumerate(pruning_idxs):
                new_weight[:, p_idx] = pruned_w_data[:, i]

        # 6) Write back the new full-size weights into the rebuilt layer
        rebuilt_layer.weight.data = new_weight
        if rebuilt_layer.bias is not None and new_bias is not None:
            rebuilt_layer.bias.data = new_bias

        print(f"Rebuilt layer {base_name} with {len(remain_indices)} remaining channels and {len(pruning_idxs)} pruned channels (dim={dim}).")

    print("Rebuild complete!")
    return rebuilt_model