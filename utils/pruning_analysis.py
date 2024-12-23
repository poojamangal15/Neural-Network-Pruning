import torch
import torch.nn as nn
def get_pruned_indices_and_counts(original_model, pruned_channels_dict):
    """
    Retrieve actual pruned channels from the *original* model
    using the pruned_channels_dict we recorded at pruning time.

    Returns:
        pruned_info: dict, { layer_name: { "pruned_dim0": [...], "pruned_dim1": [] } }
        num_pruned_channels: dict, { layer_name: (num_pruned_out, 0) }
        pruned_weights: dict, { layer_name: torch.Tensor of shape [num_pruned_out, inC, ...] }
    """
    import torch.nn as nn

    pruned_info = {}
    num_pruned_channels = {}
    pruned_weights = {}

    # We'll iterate over the original_model's modules,
    # and see if we have pruned channels for that layer in pruned_channels_dict
    for name, layer in original_model.named_modules():
        print(f"Layer {name} found in pruned_channels_dict")
        if name in pruned_channels_dict:
            # print(f"Layer {name} found in pruned_channels_dict")

            out_idxs = pruned_channels_dict[name]
            if isinstance(layer, nn.Conv2d):
                # For a conv layer, weight shape is [outC, inC, kH, kW]
                with torch.no_grad():
                    original_w = layer.weight.detach().clone()
                    # Slicing the original's pruned channels
                    pruned_w = original_w[out_idxs, :, :, :]

                # Fill the results
                pruned_info[name] = {
                    "pruned_dim0": out_idxs,
                    "pruned_dim1": []
                }
                num_pruned_channels[name] = (len(out_idxs), 0)
                pruned_weights[name] = pruned_w

            elif isinstance(layer, nn.Linear):
                # For a linear layer, weight shape is [out_features, in_features]
                with torch.no_grad():
                    original_w = layer.weight.detach().clone()
                    pruned_w = original_w[out_idxs, :]

                pruned_info[name] = {
                    "pruned_dim0": out_idxs,
                    "pruned_dim1": []
                }
                num_pruned_channels[name] = (len(out_idxs), 0)
                pruned_weights[name] = pruned_w

    return pruned_info, num_pruned_channels, pruned_weights



def get_unpruned_indices_and_counts(pruned_model):
    """
    Analyze the pruned model to determine:
    - Indices of unpruned channels (dim 0 and dim 1).
    - Number of unpruned channels in each layer.
    - Weights of unpruned channels for analysis.

    Parameters:
    - pruned_model: The PyTorch model after pruning.

    Returns:
    - unpruned_info: Dictionary of unpruned indices and counts for each layer.
    - num_unpruned_channels: Dictionary of the number of unpruned channels per layer.
    - unpruned_weights: Dictionary of unpruned weights for each layer.
    """
    unpruned_info = {}
    num_unpruned_channels = {}
    unpruned_weights = {}

    for name, layer in pruned_model.named_modules():
        if isinstance(layer, nn.Conv2d):
            layer_info = {}

            # Unpruned weight shapes
            weights = layer.weight.detach()
            out_channels, in_channels = weights.shape[:2]

            # Unpruned indices
            unpruned_dim0 = list(range(out_channels))  # Output channels
            unpruned_dim1 = list(range(in_channels))  # Input channels

            layer_info["unpruned_dim0"] = unpruned_dim0
            layer_info["unpruned_dim1"] = unpruned_dim1

            # Store weights of unpruned channels
            unpruned_weights[name] = weights

            unpruned_info[name] = layer_info
            num_unpruned_channels[name] = (len(unpruned_dim0), len(unpruned_dim1))

    return unpruned_info, num_unpruned_channels, unpruned_weights


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

def rebuild_model(
    original_model_class,
    pruned_model,
    pruned_channels_dict,
    original_model=None,
    checkpoint_path=None,
    device='cpu'
):
    """
    Example function to create a new model with the original architecture size,
    plugging back in pruned channels.
    """
    # 1. Create a fresh copy of the original architecture
    if original_model is not None:
        rebuilt_model = original_model_class(**original_model.hparams) if hasattr(original_model, 'hparams') else \
                        original_model_class()
        rebuilt_model.load_state_dict(original_model.state_dict(), strict=True)
    elif checkpoint_path is not None:
        # load from checkpoint
        rebuilt_model = original_model_class.load_from_checkpoint(checkpoint_path)
    else:
        # or just new instance from scratch
        rebuilt_model = original_model_class()
    rebuilt_model.to(device)
    rebuilt_model.eval()

    # 2. A utility to match layers by name
    def get_layer_by_name(model, layer_name):
        for name, layer in model.named_modules():
            if name == layer_name:
                return layer
        return None

    # 3. For each pruned layer, we copy in the "unpruned" channels from pruned_model
    #    and re-initialize or restore the pruned channels from original_model
    for (name_pruned, layer_pruned) in pruned_model.named_modules():
        if name_pruned in pruned_channels_dict:
            out_idxs_pruned = pruned_channels_dict[name_pruned]
            rebuilt_layer = get_layer_by_name(rebuilt_model, name_pruned)
            if rebuilt_layer is None:
                continue
            if isinstance(layer_pruned, nn.Conv2d):
                # unpruned channels in pruned_model => set(range(total_outC)) - set(pruned)
                outC_big = rebuilt_layer.weight.shape[0]
                all_out = set(range(outC_big))
                unpruned_out_idxs = list(all_out - set(out_idxs_pruned))

                # copy unpruned channels
                for new_idx, old_idx in zip(unpruned_out_idxs, range(len(unpruned_out_idxs))):
                    rebuilt_layer.weight.data[new_idx] = layer_pruned.weight.data[old_idx]
                if rebuilt_layer.bias is not None and layer_pruned.bias is not None:
                    for new_idx, old_idx in zip(unpruned_out_idxs, range(len(unpruned_out_idxs))):
                        rebuilt_layer.bias.data[new_idx] = layer_pruned.bias.data[old_idx]

                # re-inject pruned channels from original_model or random init
                if original_model is not None:
                    orig_layer = get_layer_by_name(original_model, name_pruned)
                    if orig_layer is not None:
                        for idx in out_idxs_pruned:
                            rebuilt_layer.weight.data[idx] = orig_layer.weight.data[idx]
                            if rebuilt_layer.bias is not None and orig_layer.bias is not None:
                                rebuilt_layer.bias.data[idx] = orig_layer.bias.data[idx]
                else:
                    # random init
                    for idx in out_idxs_pruned:
                        nn.init.kaiming_normal_(rebuilt_layer.weight.data[idx:idx+1])
                        if rebuilt_layer.bias is not None:
                            rebuilt_layer.bias.data[idx] = 0.0

            # elif isinstance(layer_pruned, nn.Linear):
            #   similar approach

        else:
            # if layer wasn't pruned, just copy everything over
            rebuilt_layer = get_layer_by_name(rebuilt_model, name_pruned)
            if rebuilt_layer and isinstance(layer_pruned, (nn.Conv2d, nn.Linear)):
                rebuilt_layer.weight.data.copy_(layer_pruned.weight.data)
                if rebuilt_layer.bias is not None and layer_pruned.bias is not None:
                    rebuilt_layer.bias.data.copy_(layer_pruned.bias.data)

    return rebuilt_model
