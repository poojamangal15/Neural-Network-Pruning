import torch
import torch_pruning as tp
import torch.nn as nn
import os
import copy
from copy import deepcopy
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    CyclicLR,
    CosineAnnealingLR,
    ReduceLROnPlateau
)
from tqdm import tqdm
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import re


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")

def verify_pruned_slices(module_dict, pruned_info, pruned_weights):
    """
    Verifies that the pruned weight slices in `pruned_weights` match
    the corresponding slices of the original model’s weights.
    """
    for layer_name, layer_indices in pruned_info.items():
        if layer_name not in pruned_weights:
            continue
        pruned_weight_tensor = pruned_weights[layer_name]
        pruned_dim0 = layer_indices['pruned_dim0']
        pruned_dim1 = layer_indices['pruned_dim1']

        original_layer = module_dict.get(layer_name, None)
        if not original_layer or not hasattr(original_layer, "weight"):
            continue

        # Original reference slice
        ref_slice = None
        w = original_layer.weight.detach()

        if pruned_dim0 and pruned_dim1:
            ref_slice = w[pruned_dim0][:, pruned_dim1, :, :]
        elif pruned_dim0:
            ref_slice = w[pruned_dim0]
        elif pruned_dim1:
            ref_slice = w[:, pruned_dim1]
        else:
            # No pruned indices => skip
            continue

        # Convert both to CPU for comparison
        pruned_w_cpu = pruned_weight_tensor.cpu()
        ref_slice_cpu = ref_slice.cpu()

        if not torch.allclose(pruned_w_cpu, ref_slice_cpu, atol=1e-6):
            print(f"[Mismatch ❌] Pruned slice for layer '{layer_name}' does NOT match the original.")
        else:
            print(f"[Match ✅] Pruned slice for layer '{layer_name}' correctly matches the original.")


def verify_unpruned_slices(module_dict, unpruned_info, unpruned_weights):
    """
    Verifies that the unpruned weight slices in `unpruned_weights` match
    the corresponding slices of the original model’s weights.
    """
    for layer_name, layer_indices in unpruned_info.items():
        if layer_name not in unpruned_weights:
            continue
        unpruned_weight_tensor = unpruned_weights[layer_name]
        unpruned_dim0 = layer_indices['unpruned_dim0']
        unpruned_dim1 = layer_indices['unpruned_dim1']

        original_layer = module_dict.get(layer_name, None)
        if not original_layer or not hasattr(original_layer, "weight"):
            continue

        # Original reference slice
        ref_slice = None
        w = original_layer.weight.detach()

        if unpruned_dim0 and unpruned_dim1:
            ref_slice = w[unpruned_dim0][:, unpruned_dim1, :, :]
        elif unpruned_dim0:
            ref_slice = w[unpruned_dim0]
        elif unpruned_dim1:
            ref_slice = w[:, unpruned_dim1]
        else:
            # No unpruned channels => skip
            continue

        unpruned_w_cpu = unpruned_weight_tensor.cpu()
        ref_slice_cpu = ref_slice.cpu()

        if not torch.allclose(unpruned_w_cpu, ref_slice_cpu, atol=1e-6):
            print(f"[Mismatch ❌] Unpruned slice for layer '{layer_name}' does NOT match the original.")
        else:
            print(f"[Match ✅] Unpruned slice for layer '{layer_name}' correctly matches the original.")


def prune_model(original_model, model, device, pruning_percentage=0.2, layer_pruning_percentages=None):
    pruned_info = {}
    
    model = model.to(device)
    example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

    DG = tp.DependencyGraph().build_dependency(model.model, example_inputs)
    layers_to_prune = {
        "features.3": model.model.features[3],
        "features.6": model.model.features[6],
        "features.8": model.model.features[8],
        "features.10": model.model.features[10],
    }


    def get_pruning_indices(module, percentage):
        with torch.no_grad():
            weight = module.weight.data
            if isinstance(module, torch.nn.Conv2d):
                channel_norms = weight.abs().mean(dim=[1,2,3])  
            else:
                return None

            pruning_count = int(channel_norms.size(0) * percentage)
            if pruning_count == 0:
                return []
            _, prune_indices = torch.topk(channel_norms, pruning_count, largest=False)
            return prune_indices.tolist()
 
    groups = []
    for layer_name, layer_module in layers_to_prune.items():
        if isinstance(layer_module, torch.nn.Conv2d):
            prune_fn = tp.prune_conv_out_channels
        else:
            print(f"Skipping {layer_name}: Unsupported layer type {type(layer_module)}")
            continue
        
        #incase of layer wise pruning
        if layer_pruning_percentages:
            pruning_percentage = layer_pruning_percentages.get(layer_name, 0.2)     # Default to 20% if not specified
            
        pruning_idxs = get_pruning_indices(layer_module, pruning_percentage)
        if pruning_idxs is None or len(pruning_idxs) == 0:
            print(f"No channels to prune for {layer_name}.")
            continue

        group = DG.get_pruning_group(layer_module, prune_fn, idxs=pruning_idxs)
        # print("group.details()", group.details()) 
        if DG.check_pruning_group(group):
            groups.append((layer_name, group))
        else:
            print(f"Invalid pruning group for layer {layer_name}, skipping pruning.")

    if groups:
        for layer_name, group in groups:
            print(f"Pruning layer: {layer_name}")
            group.prune()

        print("MODEL AFTER PRUNING:\n", model.model)
    else:
        print("No valid pruning groups found. The model was not pruned.")

    # Check for all the pruned and unpruned indices and weights    
    pruned_info, num_pruned_channels, pruned_weights = get_pruned_info(groups, original_model, layers_to_prune)
    unpruned_info, num_unpruned_channels, unpruned_weights = get_unpruned_info(groups, original_model, pruned_info)

    # verify_pruned_slices(dict(original_model.named_modules()), pruned_info, pruned_weights)
    # verify_unpruned_slices(dict(original_model.named_modules()), unpruned_info, unpruned_weights)

    pruned_and_unpruned_info = {"pruned_info": pruned_info, 
                                "num_pruned_channels": num_pruned_channels, 
                                "pruned_weights": pruned_weights, 
                                "unpruned_info": unpruned_info, 
                                "num_unpruned_channels": num_unpruned_channels, 
                                "unpruned_weights": unpruned_weights}
    return model, pruned_and_unpruned_info

def get_pruned_info(groups, original_model, layers_to_prune):
    """
    Collect pruned indices and total counts for each layer in the model.

    Parameters:
    - groups: List of pruning groups obtained during pruning (from torch_pruning or similar).
    - original_model: The original, unpruned model.
    - layers_to_prune: Dictionary of layers selected for pruning.

    Returns:
    - pruned_info: A dictionary mapping layer_name -> {'pruned_dim0': [...], 'pruned_dim1': [...]}.
    - num_pruned_channels: A dictionary mapping layer_name -> (num_pruned_out, num_pruned_in).
    - pruned_weights: A dictionary mapping layer_name -> tensor of pruned weights for that layer.
    """

    pruned_info = {}
    num_pruned_channels = {}
    pruned_weights = {}

    module_dict = dict(original_model.named_modules())

    all_conv2d_channels = ["features.3", "features.6", "features.8", "features.10"]

    # Global storage to track pruned indices across ALL layers
    global_pruned_info = {layer_name: {'pruned_dim0': [], 'pruned_dim1': []} for layer_name in all_conv2d_channels}

    # Regular expression to extract layer names (after `=>`)
    pattern = re.compile(r"=> prune_(out|in)_channels on ([\w\d\.\-]+)")

    # First pass: Collect all pruning indices across layers
    for layer_name, group in groups:
        if layer_name not in layers_to_prune:  # Skip layers not in our target pruning list
            continue

        for item in group.items:
            item_str = str(item)

            # Extract the correct target layer name from `item_str`
            match = pattern.search(item_str)
            if match:
                pruning_type, target_layer_name = match.groups()  # Extract pruning type and correct target layer name

                # Ensure the extracted layer is in `layers_to_prune`
                if target_layer_name in all_conv2d_channels:
                    # print("TAGRET LAYER NAME", target_layer_name)
                    # print(f"🔍 Extracted Pruning: {pruning_type} on {target_layer_name}, Indices: {item.idxs}")

                    if pruning_type == "out":
                        global_pruned_info[target_layer_name]['pruned_dim0'].extend(item.idxs)
                    elif pruning_type == "in":
                        global_pruned_info[target_layer_name]['pruned_dim1'].extend(item.idxs)

    # Second pass: Apply pruning information to layers
    for layer_name, layer in module_dict.items():
        if layer_name not in all_conv2d_channels:  # Skip layers that aren't in layers_to_prune
            continue

        if not isinstance(layer, nn.Conv2d):
            continue  # Ensure only Conv2D layers are considered

        out_channels, in_channels = layer.weight.shape[:2]

        # Get the final pruned indices (ensure they are within bounds)
        pruned_dim0 = sorted(idx for idx in global_pruned_info[layer_name]['pruned_dim0'] if 0 <= idx < out_channels)
        pruned_dim1 = sorted(idx for idx in global_pruned_info[layer_name]['pruned_dim1'] if 0 <= idx < in_channels)

        # Store in pruned_info
        pruned_info[layer_name] = {'pruned_dim0': pruned_dim0, 'pruned_dim1': pruned_dim1}
        num_pruned_channels[layer_name] = (len(pruned_dim0), len(pruned_dim1))

        # Extract pruned weights safely
        if pruned_dim0 or pruned_dim1:
            weights = layer.weight.detach()
            if pruned_dim0 and pruned_dim1:
                pruned_weight_tensor = weights[pruned_dim0][:, pruned_dim1, :, :]
            elif pruned_dim0:
                pruned_weight_tensor = weights[pruned_dim0]
            elif pruned_dim1:
                pruned_weight_tensor = weights[:, pruned_dim1]
            else:
                pruned_weight_tensor = torch.empty((0, 0))

            pruned_weights[layer_name] = pruned_weight_tensor
        else:
            pruned_weights[layer_name] = torch.empty((0, 0))

    return pruned_info, num_pruned_channels, pruned_weights


def get_unpruned_info(groups, original_model, pruned_info):
    
    unpruned_info = {}
    num_unpruned_channels = {}
    num_pruned_channels = {}
    unpruned_weights = {}

    module_dict = dict(original_model.named_modules())

    for layer_name, group in groups:
        # print(f"\nDEBUG: Processing group layer_name={layer_name}")

        layer = module_dict.get(layer_name, None)

        if layer is None and layer_name.startswith("model."):
            # Remove the prefix and try again
            layer_name = layer_name[len("model."):]  # Reassign directly
            layer = module_dict.get(layer_name, None)

        if layer is None:
            print(f"  -> Could not find layer '{layer_name}' in module_dict. Skipping.")
            continue
        
        # print(f"  -> Found layer '{layer_name}' in original_model: {layer}")

        # Initialize unpruned_info if this is the first time we see this layer
        if layer_name not in unpruned_info:
            unpruned_info[layer_name] = {'unpruned_dim0': [], 'unpruned_dim1': []}

        if isinstance(layer, torch.nn.Conv2d):
            total_output_channels = layer.weight.shape[0]
            total_input_channels  = layer.weight.shape[1]
            # print(f"      [DEBUG] out_ch={total_output_channels}, in_ch={total_input_channels}")

        # Build all_output_indices and all_input_indices
        all_output_indices = list(range(total_output_channels)) if total_output_channels else []
        all_input_indices  = list(range(total_input_channels))  if total_input_channels else []

        pruned_dim0 = pruned_info[layer_name]['pruned_dim0']
        pruned_dim1 = pruned_info[layer_name]['pruned_dim1']

        # Compute the unpruned
        unpruned_dim0 = [i for i in all_output_indices if i not in pruned_dim0]
        unpruned_dim1 = [i for i in all_input_indices  if i not in pruned_dim1]
        
        unpruned_info[layer_name]['unpruned_dim0'] = unpruned_dim0
        unpruned_info[layer_name]['unpruned_dim1'] = unpruned_dim1
        num_pruned_channels[layer_name] = (len(pruned_dim0), len(pruned_dim1))
        num_unpruned_channels[layer_name] = (len(unpruned_dim0), len(unpruned_dim1))

        # Extract unpruned weights
        if isinstance(layer, torch.nn.Conv2d):
            weights = layer.weight.detach().cpu()
            if unpruned_dim0 and unpruned_dim1:
                unpruned_weights[layer_name] = weights[unpruned_dim0][:, unpruned_dim1, :, :]
            else:
                print(f"    -> No unpruned channels for {layer_name}")
                unpruned_weights[layer_name] = torch.empty((0,0))
        else:
            print(f"    -> Not a Conv2d or Linear, skipping weight extraction for {layer_name}")
            unpruned_weights[layer_name] = None

    print("\nnum pruned channels:", num_pruned_channels)
    print("\nnum_unpruned_channels:", num_unpruned_channels)

    return unpruned_info, num_unpruned_channels, unpruned_weights
    
    
def soft_pruning(original_model, model, device, pruning_percentage=0.2, layer_pruning_percentages=None):
    """
    Prunes the model using Torch-Pruning's Magnitude Pruner.
    
    Parameters:
        - model (nn.Module): The PyTorch model to be pruned.
        - example_inputs (torch.Tensor): Sample input for pruning.
        - pruning_ratio (float): Percentage of channels to prune.
        - iterative_steps (int): Number of iterative pruning steps.
        
    Returns:
        - pruned_model (nn.Module): The pruned PyTorch model.
        - pruned_info (dict): Dictionary containing pruned indices.
    """
    example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)
    # Initialize Importance & Pruner
    imp = tp.importance.MagnitudeImportance(p=2)

    ignored_layers = []

    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)

    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        global_pruning=True,
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_percentage, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        
        # Soft Pruning
        for group in pruner.step(interactive=True):
            for dep, idxs in group:
                target_layer = dep.target.module
                pruning_fn = dep.handler
                if pruning_fn in [tp.prune_conv_in_channels, tp.prune_linear_in_channels]:
                    target_layer.weight.data[:, idxs] *= 0
                elif pruning_fn in [tp.prune_conv_out_channels, tp.prune_linear_out_channels]:
                    target_layer.weight.data[idxs] *= 0
                    if target_layer.bias is not None:
                        target_layer.bias.data[idxs] *= 0
                elif pruning_fn in [tp.prune_batchnorm_out_channels]:
                    target_layer.weight.data[idxs] *= 0
                    target_layer.bias.data[idxs] *= 0
                # group.prune() # <= disable hard pruning
        # print(model.conv1.weight)

        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(model)
        print(model(example_inputs).shape)
        print(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
        )
        print(
            "  Iter %d/%d, MACs: %.2f G => %.2f G"
            % (i+1, iterative_steps, base_macs / 1e9, macs / 1e9)
        )
        count_zero_weights(model)
        # get_nonzero_indices(model)

    # Check for all the pruned and unpruned indices and weights    
    pruned_info, num_pruned_channels, pruned_weights = get_pruned_indices_and_counts(model)
    unpruned_info, num_unpruned_channels, unpruned_weights = get_unpruned_indices_and_counts(model)

    pruned_and_unpruned_info = {"pruned_info": pruned_info, 
                                "num_pruned_channels": num_pruned_channels, 
                                "pruned_weights": pruned_weights, 
                                "unpruned_info": unpruned_info, 
                                "num_unpruned_channels": num_unpruned_channels, 
                                "unpruned_weights": unpruned_weights}
        # finetune your model here
    return model, pruned_and_unpruned_info

def high_level_pruner_old(original_model, model, device, pruning_percentage=0.2, layer_pruning_percentages=None):
    """
    Prunes the model using Torch-Pruning's Magnitude Pruner.
    
    Parameters:
        - model (nn.Module): The PyTorch model to be pruned.
        - example_inputs (torch.Tensor): Sample input for pruning.
        - pruning_ratio (float): Percentage of channels to prune.
        - iterative_steps (int): Number of iterative pruning steps.
        
    Returns:
        - pruned_model (nn.Module): The pruned PyTorch model.
        - pruned_info (dict): Dictionary containing pruned indices.
    """
    example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)
    # Initialize Importance & Pruner
    imp = tp.importance.MagnitudeImportance(p=2)

    ignored_layers = []

    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)

    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        # global_pruning=True,
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_percentage,
        ignored_layers=ignored_layers,
    )

    # Dictionary to store formatted pruning info
    pruned_weights = {}
    num_pruned_channels = {}
    pruned_info = {}

    for i in range(iterative_steps):
        for group in pruner.step(interactive=True):
            for dep, idxs in group:
                target_layer = dep.target.module
                pruning_fn = dep.handler

                # Get layer name
                layer_name = None
                for name, module in model.named_modules():
                    if module is target_layer:
                        layer_name = name
                        break
                
                if layer_name is None:
                    continue  # Skip if layer name is not found

                if layer_name.startswith("model."):
                    layer_name = layer_name[len("model."):]  # Reassign directly

                # Initialize storage format
                if layer_name not in pruned_info and isinstance(target_layer, nn.Conv2d):
                    pruned_info[layer_name] = {'pruned_dim0': [], 'pruned_dim1': []}
                    pruned_weights[layer_name] = torch.empty(0)
                    num_pruned_channels[layer_name] = (0, 0)

                if isinstance(target_layer, nn.Conv2d):
                    if pruning_fn in [tp.prune_conv_in_channels]:
                        pruned_info[layer_name]['pruned_dim1'].extend(idxs)
                        num_pruned_channels[layer_name] = (num_pruned_channels[layer_name][0], len(pruned_info[layer_name]['pruned_dim1']))
                        pruned_weights[layer_name] = target_layer.weight.data[:, idxs, :, :].clone()


                    elif pruning_fn in [tp.prune_conv_out_channels]:
                        pruned_info[layer_name]['pruned_dim0'].extend(idxs)
                        num_pruned_channels[layer_name] = (len(pruned_info[layer_name]['pruned_dim0']), num_pruned_channels[layer_name][1])
                        pruned_weights[layer_name] = target_layer.weight.data[idxs, :, :, :].clone()

                # elif isinstance(target_layer, nn.Linear):
                #     if pruning_fn in [tp.prune_linear_in_channels]:
                #         pruned_info[layer_name]['pruned_dim1'].extend(idxs)
                #         num_pruned_channels[layer_name] = (num_pruned_channels[layer_name][0], len(pruned_info[layer_name]['pruned_dim1']))
                #         pruned_weights[layer_name] = target_layer.weight.data[:, idxs].clone()


                #     elif pruning_fn in [tp.prune_linear_out_channels]:
                #         pruned_info[layer_name]['pruned_dim0'].extend(idxs)
                #         num_pruned_channels[layer_name] = (len(pruned_info[layer_name]['pruned_dim0']), num_pruned_channels[layer_name][1])
                #         pruned_weights[layer_name] = target_layer.weight.data[idxs, :].clone()
                
            group.prune()    

        print("num pruned info", num_pruned_channels)
    unpruned_info, num_unpruned_channels, unpruned_weights = get_unpruned_info_high_level(model, pruned_info)

    module_dict = dict(original_model.named_modules())
    verify_pruned_slices(module_dict, pruned_info, pruned_weights)
    verify_unpruned_slices(module_dict, unpruned_info, unpruned_weights)

    pruned_and_unpruned_info = {"pruned_info": pruned_info, 
                                "num_pruned_channels": num_pruned_channels, 
                                "pruned_weights": pruned_weights, 
                                "unpruned_info": unpruned_info, 
                                "num_unpruned_channels": num_unpruned_channels, 
                                "unpruned_weights": unpruned_weights}
        # finetune your model here
    return model, pruned_and_unpruned_info


def high_level_pruner(original_model, model, device, pruning_percentage=0.2):
    """
    Prunes 'model' in-place using Torch-Pruning's MagnitudePruner,
    but slices final pruned/unpruned weights from 'original_model'
    for verification and storage.
    """
    example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)
    imp = tp.importance.MagnitudeImportance(p=2)

    # Identify layers to ignore (final linear, etc.)
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)

    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_percentage,
        global_pruning=True,
        ignored_layers=ignored_layers,
    )

    # We'll store final pruned indices
    # pruned_info[layer_name] = {'pruned_dim0': [...], 'pruned_dim1': [...]}
    pruned_info = {}
    num_pruned_channels = {}

    # ---------------------------
    # 1) PRUNE & COLLECT PRUNED INDICES
    # ---------------------------
    for step_id in range(iterative_steps):
        for group in pruner.step(interactive=True):
            for dep, idxs in group:
                target_layer = dep.target.module
                pruning_fn = dep.handler

                # Identify the layer name
                layer_name = None
                for n, mod in model.named_modules():
                    if mod is target_layer:
                        layer_name = n
                        break
                if layer_name is None:
                    continue

                if layer_name.startswith("model."):
                    layer_name = layer_name[len("model."):]

                if not isinstance(target_layer, nn.Conv2d):
                    continue

                if layer_name not in pruned_info:
                    pruned_info[layer_name] = {'pruned_dim0': [], 'pruned_dim1': []}
                    num_pruned_channels[layer_name] = (0, 0)

                if pruning_fn in [tp.prune_conv_out_channels]:
                    pruned_info[layer_name]['pruned_dim0'].extend(idxs)
                elif pruning_fn in [tp.prune_conv_in_channels]:
                    pruned_info[layer_name]['pruned_dim1'].extend(idxs)

            group.prune()

    # ---------------------------
    # 2) DEDUPLICATE PRUNED INDICES
    # ---------------------------
    module_dict = dict(original_model.named_modules())
    for ln, dims_dict in pruned_info.items():
        dims_dict['pruned_dim0'] = sorted(set(dims_dict['pruned_dim0']))
        dims_dict['pruned_dim1'] = sorted(set(dims_dict['pruned_dim1']))
        num_pruned_channels[ln] = (
            len(dims_dict['pruned_dim0']),
            len(dims_dict['pruned_dim1'])
        )

    print("num pruned info:", num_pruned_channels)

    # ---------------------------
    # 3) DERIVE UNPRUNED INDICES FROM THE ORIGINAL MODEL
    # ---------------------------
    unpruned_info = {}
    for layer_name, dims_dict in pruned_info.items():
        orig_layer = module_dict.get(layer_name)
        if not orig_layer or not isinstance(orig_layer, nn.Conv2d):
            continue

        out_channels = orig_layer.weight.shape[0]
        in_channels = orig_layer.weight.shape[1]

        pruned_out = set(dims_dict['pruned_dim0'])
        pruned_in = set(dims_dict['pruned_dim1'])

        all_out = set(range(out_channels))
        all_in  = set(range(in_channels))

        unpruned_out = sorted(all_out - pruned_out)
        unpruned_in  = sorted(all_in  - pruned_in)

        unpruned_info[layer_name] = {
            'unpruned_dim0': unpruned_out,
            'unpruned_dim1': unpruned_in
        }

    # For any Conv2d not present in pruned_info (no channels pruned),
    # we might want to store them in unpruned_info as well:
    for name, mod in module_dict.items():
        if isinstance(mod, nn.Conv2d) and name not in pruned_info:
            out_channels = mod.weight.shape[0]
            in_channels  = mod.weight.shape[1]
            unpruned_info[name] = {
                'unpruned_dim0': list(range(out_channels)),
                'unpruned_dim1': list(range(in_channels))
            }
            num_pruned_channels[name] = (0,0)  # No pruned channels

    # Build num_unpruned_channels
    num_unpruned_channels = {}
    for ln, dims_dict in unpruned_info.items():
        num_out = len(dims_dict['unpruned_dim0'])
        num_in  = len(dims_dict['unpruned_dim1'])
        num_unpruned_channels[ln] = (num_out, num_in)

    # ---------------------------
    # 4) SLICE PRUNED & UNPRUNED WEIGHTS FROM ORIGINAL MODEL
    # ---------------------------
    pruned_weights = {}
    unpruned_weights = {}

    for layer_name in sorted(set(list(pruned_info.keys()) + list(unpruned_info.keys()))):
        layer = module_dict.get(layer_name, None)
        if not layer or not isinstance(layer, nn.Conv2d):
            continue

        w = layer.weight.detach()

        # Pruned
        pd0 = pruned_info.get(layer_name, {}).get('pruned_dim0', [])
        pd1 = pruned_info.get(layer_name, {}).get('pruned_dim1', [])
        if pd0 or pd1:
            if pd0 and pd1:
                pruned_weights[layer_name] = w[pd0][:, pd1, :, :].clone()
            elif pd0:
                pruned_weights[layer_name] = w[pd0].clone()
            elif pd1:
                pruned_weights[layer_name] = w[:, pd1].clone()
            else:
                pruned_weights[layer_name] = torch.empty((0,))
        else:
            # no pruned channels
            pruned_weights[layer_name] = torch.empty((0,))

        # Unpruned
        ud0 = unpruned_info[layer_name]['unpruned_dim0']
        ud1 = unpruned_info[layer_name]['unpruned_dim1']
        if ud0 or ud1:
            if ud0 and ud1:
                unpruned_weights[layer_name] = w[ud0][:, ud1, :, :].clone()
            elif ud0:
                unpruned_weights[layer_name] = w[ud0].clone()
            elif ud1:
                unpruned_weights[layer_name] = w[:, ud1].clone()
        else:
            unpruned_weights[layer_name] = torch.empty((0,))

    # ---------------------------
    # 5) RUN VERIFICATION
    # ---------------------------
    verify_pruned_slices(module_dict, pruned_info, pruned_weights)
    verify_unpruned_slices(module_dict, unpruned_info, unpruned_weights)

    # ---------------------------
    # 6) BUILD DICT TO RETURN
    # ---------------------------
    pruned_and_unpruned_info = {
        "pruned_info": pruned_info,
        "num_pruned_channels": num_pruned_channels,
        "pruned_weights": pruned_weights,
        "unpruned_info": unpruned_info,
        "num_unpruned_channels": num_unpruned_channels,
        "unpruned_weights": unpruned_weights
    }

    return model, pruned_and_unpruned_info


import torch
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp

def high_level_prunerTaylor(
    original_model,
    model,
    device,
    train_loader,
    pruning_percentage=0.2
):
    """
    Prunes 'model' in-place using Torch-Pruning's TaylorImportance,
    but slices final pruned/unpruned weights from 'original_model'
    for verification and storage.

    Parameters:
        original_model (nn.Module): The unpruned reference model (used for indexing and verification).
        model (nn.Module): The PyTorch model to be pruned in-place.
        device (torch.device): The computing device (CPU/MPS/CUDA).
        train_loader (DataLoader): A DataLoader to compute gradients for Taylor importance.
        pruning_percentage (float): Global percentage of channels to prune (0.0 - 1.0).

    Returns:
        (model, pruned_and_unpruned_info) tuple:
           model: The pruned model (on 'device').
           pruned_and_unpruned_info: dict containing pruned/unpruned indices, slices, etc.
    """
    # ---------------------------------------------------------------------
    # 1) Compute One Batch of Gradients for Taylor Importance
    # ---------------------------------------------------------------------
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # We'll just grab a single small batch from train_loader
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()  # Required for Taylor-based pruning
    # (We only do one batch to measure gradients)

    # ---------------------------------------------------------------------
    # 2) Build a Taylor Pruner
    # ---------------------------------------------------------------------
    example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)
    imp = tp.importance.TaylorImportance()  # uses .grad in the weights

    ignored_layers = []
    for m in model.modules():
        # e.g. if final linear has out_features=10 for a 10-class classifier, ignore it
        if isinstance(m, nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=imp,                 # <--- this is the difference: Taylor-based importance
        iterative_steps=1,              # If you want iterative pruning, increase this.
        pruning_ratio=pruning_percentage,
        ignored_layers=ignored_layers,
    )

    # We'll store final pruned indices here:
    #   pruned_info[layer_name] = {'pruned_dim0': [...], 'pruned_dim1': [...]}
    pruned_info = {}
    num_pruned_channels = {}

    # ---------------------------------------------------------------------
    # 3) Prune & Collect Indices
    # ---------------------------------------------------------------------
    # The pruner.step() returns groups to prune. We step in interactive mode just like Magnitude.
    for group in pruner.step(interactive=True):
        # Each group is a list of (dep, idxs) pairs
        for dep, idxs in group:
            target_layer = dep.target.module
            pruning_fn   = dep.handler

            # Figure out which layer name this is
            layer_name = None
            for n, mod in model.named_modules():
                if mod is target_layer:
                    layer_name = n
                    break
            if layer_name is None:
                continue  # skip if not found

            if layer_name.startswith("model."):
                layer_name = layer_name[len("model."):]

            # We only handle Conv2d for now
            if not isinstance(target_layer, nn.Conv2d):
                continue

            if layer_name not in pruned_info:
                pruned_info[layer_name] = {'pruned_dim0': [], 'pruned_dim1': []}
                num_pruned_channels[layer_name] = (0, 0)

            # If out_channels are pruned
            if pruning_fn in [tp.prune_conv_out_channels]:
                pruned_info[layer_name]['pruned_dim0'].extend(idxs)
            # If in_channels are pruned
            elif pruning_fn in [tp.prune_conv_in_channels]:
                pruned_info[layer_name]['pruned_dim1'].extend(idxs)

        # Actually perform the prune in-place
        group.prune()

    # ---------------------------------------------------------------------
    # 4) Deduplicate & Summarize Pruned Indices
    # ---------------------------------------------------------------------
    module_dict = dict(original_model.named_modules())
    for ln, dims_dict in pruned_info.items():
        dims_dict['pruned_dim0'] = sorted(set(dims_dict['pruned_dim0']))
        dims_dict['pruned_dim1'] = sorted(set(dims_dict['pruned_dim1']))
        out_pruned_count = len(dims_dict['pruned_dim0'])
        in_pruned_count  = len(dims_dict['pruned_dim1'])
        num_pruned_channels[ln] = (out_pruned_count, in_pruned_count)

    # Debug info
    print("[Taylor Pruning] num_pruned_channels:", num_pruned_channels)

    # ---------------------------------------------------------------------
    # 5) Compute Unpruned Indices from the Original Model
    # ---------------------------------------------------------------------
    unpruned_info = {}
    for layer_name, dims_dict in pruned_info.items():
        orig_layer = module_dict.get(layer_name, None)
        if not orig_layer or not isinstance(orig_layer, nn.Conv2d):
            continue

        out_channels = orig_layer.weight.shape[0]
        in_channels  = orig_layer.weight.shape[1]

        pruned_out = set(dims_dict['pruned_dim0'])
        pruned_in  = set(dims_dict['pruned_dim1'])

        all_out = set(range(out_channels))
        all_in  = set(range(in_channels))

        unpruned_out = sorted(all_out - pruned_out)
        unpruned_in  = sorted(all_in  - pruned_in)

        unpruned_info[layer_name] = {
            'unpruned_dim0': unpruned_out,
            'unpruned_dim1': unpruned_in
        }

    # For any Conv2d not present in pruned_info, no channels got pruned
    for name, mod in module_dict.items():
        if isinstance(mod, nn.Conv2d) and name not in pruned_info:
            out_channels = mod.weight.shape[0]
            in_channels  = mod.weight.shape[1]
            unpruned_info[name] = {
                'unpruned_dim0': list(range(out_channels)),
                'unpruned_dim1': list(range(in_channels))
            }
            num_pruned_channels[name] = (0, 0)

    num_unpruned_channels = {}
    for ln, dims_dict in unpruned_info.items():
        num_out = len(dims_dict['unpruned_dim0'])
        num_in  = len(dims_dict['unpruned_dim1'])
        num_unpruned_channels[ln] = (num_out, num_in)

    # ---------------------------------------------------------------------
    # 6) Slice Pruned & Unpruned Weights from the Original Model
    # ---------------------------------------------------------------------
    pruned_weights = {}
    unpruned_weights = {}

    for layer_name in sorted(set(list(pruned_info.keys()) + list(unpruned_info.keys()))):
        layer = module_dict.get(layer_name, None)
        if not layer or not isinstance(layer, nn.Conv2d):
            continue

        w = layer.weight.detach()

        # PRUNED
        pd0 = pruned_info.get(layer_name, {}).get('pruned_dim0', [])
        pd1 = pruned_info.get(layer_name, {}).get('pruned_dim1', [])
        if pd0 or pd1:
            if pd0 and pd1:
                pruned_weights[layer_name] = w[pd0][:, pd1, :, :].clone()
            elif pd0:
                pruned_weights[layer_name] = w[pd0].clone()
            elif pd1:
                pruned_weights[layer_name] = w[:, pd1].clone()
            else:
                pruned_weights[layer_name] = torch.empty((0,))
        else:
            pruned_weights[layer_name] = torch.empty((0,))

        # UNPRUNED
        ud0 = unpruned_info[layer_name]['unpruned_dim0']
        ud1 = unpruned_info[layer_name]['unpruned_dim1']
        if ud0 or ud1:
            if ud0 and ud1:
                unpruned_weights[layer_name] = w[ud0][:, ud1, :, :].clone()
            elif ud0:
                unpruned_weights[layer_name] = w[ud0].clone()
            elif ud1:
                unpruned_weights[layer_name] = w[:, ud1].clone()
            else:
                unpruned_weights[layer_name] = torch.empty((0,))
        else:
            unpruned_weights[layer_name] = torch.empty((0,))

    # ---------------------------------------------------------------------
    # 7) Verification (Optional, if you have these functions)
    # ---------------------------------------------------------------------
    # from your verification utils: verify_pruned_slices, verify_unpruned_slices
    # module_dict = dict(original_model.named_modules())
    # verify_pruned_slices(module_dict, pruned_info, pruned_weights)
    # verify_unpruned_slices(module_dict, unpruned_info, unpruned_weights)

    # ---------------------------------------------------------------------
    # 8) Build the Return Dictionary
    # ---------------------------------------------------------------------
    pruned_and_unpruned_info = {
        "pruned_info": pruned_info,
        "num_pruned_channels": num_pruned_channels,
        "pruned_weights": pruned_weights,
        "unpruned_info": unpruned_info,
        "num_unpruned_channels": num_unpruned_channels,
        "unpruned_weights": unpruned_weights
    }

    return model, pruned_and_unpruned_info


def high_level_prunerTaylor_old(original_model, model, device, train_loader, pruning_percentage=0.2, layer_pruning_percentages=None):
    """
    Prunes the model using Torch-Pruning's Taylor Importance Pruner.

    Parameters:
        original_model (nn.Module): The unpruned reference model.
        model (nn.Module): The PyTorch model to be pruned.
        device (torch.device): The computing device.
        train_loader (DataLoader): A DataLoader providing a small representative batch.
        pruning_percentage (float): Global percentage of channels to prune.
        layer_pruning_percentages (dict, optional): Layer-specific pruning ratios.
        
    Returns:
        model (nn.Module): The pruned model.
        pruned_and_unpruned_info (dict): Dictionary containing pruned and unpruned indices and weights.
    """
    # Set model to training mode to ensure gradients are computed.
    model.train()
    
    # Use a representative batch from training data to compute gradients.
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam is recommended for stable, adaptive updates.
    criterion = nn.CrossEntropyLoss()

    # Get one small representative batch.
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()  # Compute gradients required for Taylor importance.
        break  # Use only the first batch for importance estimation.
    
    # Use Taylor-based Importance (requires gradients).
    imp = tp.importance.TaylorImportance()

    # Define layers to ignore (e.g., final classifier layers)
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)

    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device),  # Example input; not used in gradient computation.
        importance=imp,  # Using Taylor-based importance.
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_percentage,
        ignored_layers=ignored_layers,
    )
    
    # Dictionaries to store pruning information.
    pruned_info = {}
    num_pruned_channels = {}
    pruned_weights = {}

    # Perform pruning in interactive mode.
    for group in pruner.step(interactive=True):
        for dep, idxs in group:
            target_layer = dep.target.module
            pruning_fn = dep.handler

            # Retrieve layer name.
            layer_name = None
            for name, module in model.named_modules():
                if module is target_layer:
                    layer_name = name
                    break
            if layer_name is None:
                continue  # Skip if not found.

            # Remove any prefix (e.g., "model.") if present.
            if layer_name.startswith("model."):
                layer_name = layer_name[len("model."):]

            # Initialize storage for Conv2d layers.
            if layer_name not in pruned_info and isinstance(target_layer, nn.Conv2d):
                pruned_info[layer_name] = {'pruned_dim0': [], 'pruned_dim1': []}
                pruned_weights[layer_name] = torch.empty(0)
                num_pruned_channels[layer_name] = (0, 0)

            if isinstance(target_layer, nn.Conv2d):
                if pruning_fn in [tp.prune_conv_in_channels]:
                    pruned_info[layer_name]['pruned_dim1'].extend(idxs)
                    num_pruned_channels[layer_name] = (num_pruned_channels[layer_name][0],
                                                       len(pruned_info[layer_name]['pruned_dim1']))
                    pruned_weights[layer_name] = target_layer.weight.data[:, idxs, :, :].clone()
                elif pruning_fn in [tp.prune_conv_out_channels]:
                    pruned_info[layer_name]['pruned_dim0'].extend(idxs)
                    num_pruned_channels[layer_name] = (len(pruned_info[layer_name]['pruned_dim0']),
                                                       num_pruned_channels[layer_name][1])
                    pruned_weights[layer_name] = target_layer.weight.data[idxs, :, :, :].clone()
        group.prune()

    # Retrieve unpruned information using your helper function.
    unpruned_info, num_unpruned_channels, unpruned_weights = get_unpruned_info_high_level(original_model, pruned_info)

    pruned_and_unpruned_info = {
        "pruned_info": pruned_info,
        "num_pruned_channels": num_pruned_channels,
        "pruned_weights": pruned_weights,
        "unpruned_info": unpruned_info,
        "num_unpruned_channels": num_unpruned_channels,
        "unpruned_weights": unpruned_weights
    }

    return model, pruned_and_unpruned_info

def compute_hessian_old(loss, model, scale_factor=1e3):
    """
    Computes an approximation of the Hessian for each trainable parameter in the model.
    - Skips parameters without gradients.
    - Multiplies the Hessian by a scale factor to amplify differences.
    
    Returns:
        hessian_dict: Dictionary mapping parameter names to their approximated Hessian.
    """
    hessian_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            try:
                # Compute Hessian for the current parameter using torch.autograd.functional.hessian.
                # We use allow_unused=True to avoid errors when a parameter doesn't contribute directly.
                hess_matrix = torch.autograd.functional.hessian(
                    lambda x: torch.autograd.grad(loss, [x], retain_graph=True, allow_unused=True)[0].sum()
                    if torch.autograd.grad(loss, [x], retain_graph=True, allow_unused=True)[0] is not None
                    else torch.tensor(0.0, device=loss.device),
                    param
                )
                # Amplify Hessian values to improve discrimination
                hess_matrix = hess_matrix * scale_factor
                hessian_dict[name] = hess_matrix
            except RuntimeError as e:
                print(f"Skipping Hessian computation for {name} due to error: {e}")
                continue
    return hessian_dict

def hessian_based_pruner_old(original_model, model, device, train_loader, pruning_percentage=0.2):
    """
    Prunes the model using Hessian-based Importance.
    
    This implementation:
      - Computes Hessian approximations over a few mini-batches.
      - Uses a scale factor to amplify the Hessian scores.
      - Uses Torch-Pruning's MetaPruner for global, importance-based channel removal.
    
    Returns:
        - pruned_model: The pruned model.
        - pruned_and_unpruned_info: A dictionary containing pruned and unpruned indices and weights.
    """
    model.train()  # Ensure gradients are tracked
    example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

    # Use Hessian-based Importance
    imp = tp.importance.HessianImportance()

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)
 
    # Use MetaPruner for global, importance-based pruning
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=pruning_percentage,
        ignored_layers=ignored_layers,
    )

    # --- Step 1: Compute Hessian Approximations ---
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Use several mini-batches to obtain a more stable Hessian estimation.
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()  # Compute first-order gradients
        if i == 3:  # After 3 mini-batches, compute Hessian approximations
            hessian_info = compute_hessian(loss, model, scale_factor=1e3)
            break

    # Attach Hessian info to parameters so the pruner can use it
    for name, param in model.named_parameters():
        if name in hessian_info:
            param.hessian = hessian_info[name]

    # --- Step 2: Perform Pruning ---
    pruned_info = {}
    num_pruned_channels = {}
    pruned_weights = {}

    for group in pruner.step(interactive=True):
        for dep, idxs in group:
            target_layer = dep.target.module
            pruning_fn = dep.handler

            # Get the name of the layer
            layer_name = None
            for name, module in model.named_modules():
                if module is target_layer:
                    layer_name = name
                    break
            if layer_name is None:
                continue
            if layer_name.startswith("model."):
                layer_name = layer_name[len("model."):]

            # Initialize storage if needed (only for Conv2d)
            if layer_name not in pruned_info and isinstance(target_layer, nn.Conv2d):
                pruned_info[layer_name] = {'pruned_dim0': [], 'pruned_dim1': []}
                pruned_weights[layer_name] = torch.empty(0)
                num_pruned_channels[layer_name] = (0, 0)

            if isinstance(target_layer, nn.Conv2d):
                if pruning_fn in [tp.prune_conv_in_channels]:
                    pruned_info[layer_name]['pruned_dim1'].extend(idxs)
                    num_pruned_channels[layer_name] = (num_pruned_channels[layer_name][0],
                                                       len(pruned_info[layer_name]['pruned_dim1']))
                    pruned_weights[layer_name] = target_layer.weight.data[:, idxs, :, :].clone()
                elif pruning_fn in [tp.prune_conv_out_channels]:
                    pruned_info[layer_name]['pruned_dim0'].extend(idxs)
                    num_pruned_channels[layer_name] = (len(pruned_info[layer_name]['pruned_dim0']),
                                                       num_pruned_channels[layer_name][1])
                    pruned_weights[layer_name] = target_layer.weight.data[idxs, :, :, :].clone()
        group.prune()

    # --- Step 3: Obtain Unpruned Information ---
    unpruned_info, num_unpruned_channels, unpruned_weights = get_unpruned_info_high_level(original_model, pruned_info)

    pruned_and_unpruned_info = {
        "pruned_info": pruned_info,
        "num_pruned_channels": num_pruned_channels,
        "pruned_weights": pruned_weights,
        "unpruned_info": unpruned_info,
        "num_unpruned_channels": num_unpruned_channels,
        "unpruned_weights": unpruned_weights
    }

    return model, pruned_and_unpruned_info

def compute_hessian(loss, model, scale_factor=1e3):
    """
    Computes an approximation of the Hessian for each trainable parameter in the model.
    - Skips parameters without gradients.
    - Multiplies the Hessian by a scale factor to amplify differences.
    
    Returns:
        hessian_dict: Dictionary mapping parameter names to their approximated Hessian.
    """
    hessian_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            try:
                # Using torch.autograd.functional.hessian for the Hessian w.r.t. param
                hess_matrix = torch.autograd.functional.hessian(
                    lambda x: torch.autograd.grad(loss, [x], retain_graph=True, allow_unused=True)[0].sum()
                    if torch.autograd.grad(loss, [x], retain_graph=True, allow_unused=True)[0] is not None
                    else torch.tensor(0.0, device=loss.device),
                    param
                )
                # Scale for better discrimination
                hess_matrix = hess_matrix * scale_factor
                hessian_dict[name] = hess_matrix
            except RuntimeError as e:
                print(f"Skipping Hessian computation for {name} due to error: {e}")
                continue
    return hessian_dict


def hessian_based_pruner(
    original_model,
    model,
    device,
    train_loader,
    pruning_percentage=0.2
):
    """
    Prunes 'model' in-place using a Hessian-based Importance approach,
    but slices final pruned/unpruned weights from 'original_model' for 
    verification and storage.

    Parameters:
        original_model (nn.Module): The unpruned reference model (for indexing & verification).
        model (nn.Module): The PyTorch model to be pruned in-place.
        device (torch.device): The computing device (CPU/MPS/CUDA).
        train_loader (DataLoader): Training data for computing Hessian approximations.
        pruning_percentage (float): Global percentage of channels to prune (0.0 - 1.0).

    Returns:
        (pruned_model, pruned_and_unpruned_info) tuple:
           pruned_model: The pruned model (on 'device').
           pruned_and_unpruned_info: dict containing pruned/unpruned indices, slices, etc.
    """
    model.to(device)
    model.train()
    
    # Example input used by the pruner for shape inference
    example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)
    
    # Define Hessian-based Importance
    imp = tp.importance.HessianImportance()

    # Identify layers to ignore (e.g. final linear for classification)
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)

    # We'll create a MetaPruner for global, Hessian-based pruning
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs=example_inputs,
        importance=imp,
        pruning_ratio=pruning_percentage,
        ignored_layers=ignored_layers
    )

    # ---------------------------
    # 1) Compute Hessian Approximations
    # ---------------------------
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # We'll do a few mini-batches to accumulate a stable Hessian approximation
    hessian_info = {}
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()  # compute first-order gradients

        # After some mini-batches, compute Hessian from this 'loss'
        # If you only want to do it once, break after first iteration
        if i == 3:  
            hessian_info = compute_hessian(loss, model, scale_factor=1e3)
            break

    # Attach Hessian info to parameters so the pruner can use it
    for name, param in model.named_parameters():
        if name in hessian_info:
            param.hessian = hessian_info[name]

    # ---------------------------
    # 2) Prune & Collect Indices
    # ---------------------------
    pruned_info = {}
    num_pruned_channels = {}

    # We'll do a single step of global pruning
    for group in pruner.step(interactive=True):
        for dep, idxs in group:
            target_layer = dep.target.module
            pruning_fn   = dep.handler

            # Identify the layer name
            layer_name = None
            for n, mod in model.named_modules():
                if mod is target_layer:
                    layer_name = n
                    break
            if layer_name is None:
                continue

            if layer_name.startswith("model."):
                layer_name = layer_name[len("model."):]

            # Only handle Conv2d
            if not isinstance(target_layer, nn.Conv2d):
                continue

            if layer_name not in pruned_info:
                pruned_info[layer_name] = {'pruned_dim0': [], 'pruned_dim1': []}
                num_pruned_channels[layer_name] = (0, 0)

            if pruning_fn in [tp.prune_conv_out_channels]:
                pruned_info[layer_name]['pruned_dim0'].extend(idxs)
            elif pruning_fn in [tp.prune_conv_in_channels]:
                pruned_info[layer_name]['pruned_dim1'].extend(idxs)

        # Actually prune in-place
        group.prune()

    # ---------------------------
    # 3) Deduplicate Pruned Indices
    # ---------------------------
    module_dict = dict(original_model.named_modules())
    for ln, dims_dict in pruned_info.items():
        dims_dict['pruned_dim0'] = sorted(set(dims_dict['pruned_dim0']))
        dims_dict['pruned_dim1'] = sorted(set(dims_dict['pruned_dim1']))
        out_pruned_count = len(dims_dict['pruned_dim0'])
        in_pruned_count  = len(dims_dict['pruned_dim1'])
        num_pruned_channels[ln] = (out_pruned_count, in_pruned_count)

    print("[Hessian Pruning] num_pruned_channels:", num_pruned_channels)

    # ---------------------------
    # 4) Compute Unpruned Indices from Original Model
    # ---------------------------
    unpruned_info = {}
    for layer_name, dims_dict in pruned_info.items():
        orig_layer = module_dict.get(layer_name, None)
        if not orig_layer or not isinstance(orig_layer, nn.Conv2d):
            continue

        out_channels = orig_layer.weight.shape[0]
        in_channels  = orig_layer.weight.shape[1]

        pruned_out = set(dims_dict['pruned_dim0'])
        pruned_in  = set(dims_dict['pruned_dim1'])

        all_out = set(range(out_channels))
        all_in  = set(range(in_channels))

        unpruned_out = sorted(all_out - pruned_out)
        unpruned_in  = sorted(all_in  - pruned_in)

        unpruned_info[layer_name] = {
            'unpruned_dim0': unpruned_out,
            'unpruned_dim1': unpruned_in
        }

    # For any Conv2d not pruned at all
    for name, mod in module_dict.items():
        if isinstance(mod, nn.Conv2d) and name not in pruned_info:
            out_channels = mod.weight.shape[0]
            in_channels  = mod.weight.shape[1]
            unpruned_info[name] = {
                'unpruned_dim0': list(range(out_channels)),
                'unpruned_dim1': list(range(in_channels))
            }
            num_pruned_channels[name] = (0, 0)

    # Build num_unpruned_channels
    num_unpruned_channels = {}
    for ln, dims_dict in unpruned_info.items():
        num_out = len(dims_dict['unpruned_dim0'])
        num_in  = len(dims_dict['unpruned_dim1'])
        num_unpruned_channels[ln] = (num_out, num_in)

    # ---------------------------
    # 5) Slice Pruned & Unpruned Weights from Original Model
    # ---------------------------
    pruned_weights = {}
    unpruned_weights = {}

    for layer_name in sorted(set(list(pruned_info.keys()) + list(unpruned_info.keys()))):
        layer = module_dict.get(layer_name, None)
        if not layer or not isinstance(layer, nn.Conv2d):
            continue

        w = layer.weight.detach()

        # Pruned
        pd0 = pruned_info.get(layer_name, {}).get('pruned_dim0', [])
        pd1 = pruned_info.get(layer_name, {}).get('pruned_dim1', [])
        if pd0 or pd1:
            if pd0 and pd1:
                pruned_weights[layer_name] = w[pd0][:, pd1, :, :].clone()
            elif pd0:
                pruned_weights[layer_name] = w[pd0].clone()
            elif pd1:
                pruned_weights[layer_name] = w[:, pd1].clone()
            else:
                pruned_weights[layer_name] = torch.empty((0,))
        else:
            pruned_weights[layer_name] = torch.empty((0,))

        # Unpruned
        ud0 = unpruned_info[layer_name]['unpruned_dim0']
        ud1 = unpruned_info[layer_name]['unpruned_dim1']
        if ud0 or ud1:
            if ud0 and ud1:
                unpruned_weights[layer_name] = w[ud0][:, ud1, :, :].clone()
            elif ud0:
                unpruned_weights[layer_name] = w[ud0].clone()
            elif ud1:
                unpruned_weights[layer_name] = w[:, ud1].clone()
            else:
                unpruned_weights[layer_name] = torch.empty((0,))
        else:
            unpruned_weights[layer_name] = torch.empty((0,))

    # ---------------------------
    # 6) (Optional) Verification
    # ---------------------------
    # from your code: verify_pruned_slices, verify_unpruned_slices
    # verify_pruned_slices(module_dict, pruned_info, pruned_weights)
    # verify_unpruned_slices(module_dict, unpruned_info, unpruned_weights)

    # ---------------------------
    # 7) Return Results
    # ---------------------------
    pruned_and_unpruned_info = {
        "pruned_info": pruned_info,
        "num_pruned_channels": num_pruned_channels,
        "pruned_weights": pruned_weights,
        "unpruned_info": unpruned_info,
        "num_unpruned_channels": num_unpruned_channels,
        "unpruned_weights": unpruned_weights
    }

    return model, pruned_and_unpruned_info


def get_unpruned_info_high_level(model, pruned_info):
    """
    Extracts unpruned indices, weights, and counts for all layers in the model.

    Parameters:
        - model (nn.Module): The PyTorch model (after soft pruning).
        - pruned_info (dict): Dictionary containing pruned indices for each layer.

    Returns:
        - unpruned_info (dict): Layer-wise unpruned indices for input & output channels.
        - num_unpruned_channels (dict): Number of remaining (unpruned) channels per layer.
        - unpruned_weights (dict): Tensor weights for unpruned channels.
    """
    unpruned_info = {}
    num_unpruned_channels = {}
    unpruned_weights = {}

    for name, module in model.named_modules():
        
        if name.startswith("model."):
            name = name[len("model."):]  # Reassign directly

        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            # Get original dimensions
            out_channels = module.weight.shape[0]
            in_channels = module.weight.shape[1] if isinstance(module, torch.nn.Conv2d) else module.weight.shape[1]

            # Get pruned indices for this layer
            pruned_out = set(pruned_info.get(name, {}).get('pruned_dim0', []))
            pruned_in = set(pruned_info.get(name, {}).get('pruned_dim1', []))

            # Compute unpruned indices
            unpruned_out = sorted(set(range(out_channels)) - pruned_out)
            unpruned_in = sorted(set(range(in_channels)) - pruned_in)

            # Store results
            unpruned_info[name] = {'unpruned_dim0': unpruned_out, 'unpruned_dim1': unpruned_in}
            num_unpruned_channels[name] = (len(unpruned_out), len(unpruned_in))

            # Extract weights for unpruned indices
            if isinstance(module, torch.nn.Conv2d):
                unpruned_weights[name] = module.weight.data[unpruned_out][:, unpruned_in, :, :].clone()
            elif isinstance(module, torch.nn.Linear):
                unpruned_weights[name] = module.weight.data[unpruned_out][:, unpruned_in].clone()

    print("num unpruned channels", num_unpruned_channels)
    return unpruned_info, num_unpruned_channels, unpruned_weights

# def get_unpruned_info_high_level(model, pruned_info):
#     """
#     Gathers unpruned channel indices based on the final shape of each (Conv2d) 
#     layer in `model` plus knowledge of which indices got pruned.

#     Returns:
#         unpruned_info[layer_name] = {
#             'unpruned_dim0': [...],
#             'unpruned_dim1': [...]
#         }
#         num_unpruned_channels[layer_name] = (#out_channels, #in_channels)
#         unpruned_weights[layer_name] = empty(0) [We won't store real unpruned slices here!]
#     """
#     import torch

#     unpruned_info = {}
#     num_unpruned_channels = {}
#     unpruned_weights = {}  # We can fill with empty placeholders for now

#     for name, module in model.named_modules():
#         if name.startswith("model."):
#             name = name[len("model."):]

#         if not isinstance(module, torch.nn.Conv2d):
#             continue

#         out_channels = module.weight.shape[0]
#         in_channels = module.weight.shape[1]

#         pruned_dim0 = set(pruned_info.get(name, {}).get('pruned_dim0', []))
#         pruned_dim1 = set(pruned_info.get(name, {}).get('pruned_dim1', []))

#         all_out = set(range(out_channels + len(pruned_dim0)))  # total original indices
#         all_in = set(range(in_channels + len(pruned_dim1)))

#         # Actually, the pruned model shape is smaller, so we have to be consistent
#         # with the original index sets. We rely on the fact that pruned_info used 
#         # original indexing. So let's define:
#         # unpruned_out = all_out - pruned_out
#         # But the "all_out" must be the original size from the original model,
#         # not from the pruned model. We'll correct that soon in the main function 
#         # by referencing the original shape. For now, let's skip it.

#         # We'll store placeholders:
#         unpruned_info[name] = {
#             'unpruned_dim0': [],  # will be set after we gather them from original
#             'unpruned_dim1': []
#         }
#         num_unpruned_channels[name] = (0, 0)
#         unpruned_weights[name] = torch.empty(0)

#     return unpruned_info, num_unpruned_channels, unpruned_weights


def get_pruned_indices_and_counts(model):
    """
    Analyze a structurally pruned model to determine:
    - The indices of pruned channels (dim 0 and dim 1).
    - The number of pruned channels in each layer.
    
    Parameters:
    - model: The PyTorch model.

    Returns:
    - pruned_info: A dictionary containing pruned indices and counts for each layer.
    """
    pruned_info = {}
    pruned_weights = {}
    num_pruned_channels = {}
    
    for name, layer in model.named_modules():
        if name.startswith("model."):
            name = name[len("model."):]

        if isinstance(layer, nn.Conv2d):  # Check for Conv2d layers
            layer_info = {}
            
            # Get the weight tensor
            weights = layer.weight.detach()
            
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
            
            # Add layer information to pruned_info
            pruned_info[name] = layer_info
            num_pruned_channels[name] = (len(pruned_dim0), len(pruned_dim1))
            
            # Save pruned weights
            if pruned_dim0 and pruned_dim1:
                pruned_weights[name] = weights[pruned_dim0, :, :, :][:, pruned_dim1, :, :]
            else:
                pruned_weights[name] = weights[pruned_dim0, :, :, :]
    
    # print(" pruned info", pruned_info)
    print("num pruned channels", num_pruned_channels)
    return pruned_info, num_pruned_channels, pruned_weights

def get_unpruned_indices_and_counts(model):
    non_pruned_info = {}
    num_unpruned_channels = {}
    unpruned_weights = {}
    
    for name, module in model.named_modules():
        if name.startswith("model."):
            name = name[len("model."):]

        if isinstance(module, nn.Conv2d):
            
            layer_info = {}
            
            # Get the weight tensor
            weights = module.weight.detach()
            
            # Identify pruned filters along dim=0 (output channels)
            pruned_dim0 = torch.nonzero(weights.abs().sum(dim=(1, 2, 3)) > 0).squeeze().tolist()
            if isinstance(pruned_dim0, int):  # Handle single index case
                pruned_dim0 = [pruned_dim0]
            layer_info['unpruned_dim0'] = pruned_dim0
            
            
            # Identify pruned filters along dim=1 (input channels)
            pruned_dim1 = torch.nonzero(weights.abs().sum(dim=(0, 2, 3)) > 0).squeeze().tolist()
            if isinstance(pruned_dim1, int):  # Handle single index case
                pruned_dim1 = [pruned_dim1]
            layer_info['unpruned_dim1'] = pruned_dim1
        
            non_pruned_info[name] = layer_info
            num_unpruned_channels[name] = (len(pruned_dim0), len(pruned_dim1))
            unpruned_weights[name] = weights[pruned_dim0][:,pruned_dim1,:,:]
                            
    # print("non pruned info", non_pruned_info)
    print("num pruned channels", num_unpruned_channels)

    return non_pruned_info, num_unpruned_channels, unpruned_weights


def count_zero_weights(model):
    total_weights = 0
    zero_weights = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_weights += param.numel()
            zero_weights += (param == 0).sum().item()
    print(f"Total weights: {total_weights}")
    print(f"Zero weights: {zero_weights}")
    print(f"Percentage of pruned weights: {100 * zero_weights / total_weights:.2f}%")

def get_nonzero_indices(model):
    non_zero_indices = {}
    new_channels = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            
            weights = module.weight.data
            indices = []

            # Iterate over both dimensions (dim 0: output channels, dim 1: input channels)
            for out_idx in range(weights.shape[0]):  # dim 0 (output channels)
                for in_idx in range(weights.shape[1]):  # dim 1 (input channels)
                    
                    if weights[out_idx, in_idx].abs().sum() > 0:  # Check if the (out_idx, in_idx) slice is non-zero
                        indices.append((out_idx, in_idx))
        
            non_zero_indices[name] = indices
            out_channels = len(set(t[0] for t in indices))
            in_channels = len(set(t[1] for t in indices))
            new_channels[name] = (out_channels, in_channels)
                            
    return non_zero_indices, new_channels

def extend_channels(model, pruned_dict):
    """
    Extend channel dictionary to include all layers, especially the first layer.
    """
    new_channel_dict = {}

    for name, module in model.named_modules():
        if name.startswith("model."):
            name = name[len("model."):]

        if isinstance(module, nn.Conv2d):

            new_in_channel = module.weight.data.shape[1] + pruned_dict.get(name, (0, 0))[1]
            new_out_channel = module.weight.data.shape[0] + pruned_dict.get(name, (0, 0))[0]

            new_channel_dict[name] = (int(new_out_channel), int(new_in_channel))

    return new_channel_dict

def get_rebuild_channels(unpruned_channels, pruned_channels):
    new_channels_dict = {}
    for name, weight in pruned_channels.items():
        new_in_channels = int(unpruned_channels[name][1] + pruned_channels[name][1])
        new_out_channels = int(unpruned_channels[name][0] + pruned_channels[name][0])
        
        new_channels_dict[name] = (new_in_channels, new_out_channels)
    
    if "features.0" not in new_channels_dict:
        new_channels_dict["features.0"] = (64, 3)

    print("REBUILD CHANNELS DICTIONARY", new_channels_dict)
    return new_channels_dict

def get_core_weights(pruned_model, unpruned_weights):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            print("UNPRUNED WEIGHTS NAME", name)
            unpruned_weights[name] = module.weight.data

class AlexNet_general(nn.Module):
    def __init__(self, channel_dict, last_conv_out_features):
        super(AlexNet_general, self).__init__()
        
        # Define features (Convolutional layers only)
        self.features = nn.Sequential(
            nn.Conv2d(channel_dict['features.0'][1], channel_dict['features.0'][0], kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(channel_dict['features.3'][1], channel_dict['features.3'][0], kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(channel_dict['features.6'][1], channel_dict['features.6'][0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(channel_dict['features.8'][1], channel_dict['features.8'][0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(channel_dict['features.10'][1], channel_dict['features.10'][0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_out_features, 4096),  # Adapted `in_features`
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Linear(4096, 10),  # Output layer (10 classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNet_general_core(nn.Module):
    def __init__(self, channel_dict):
        super(AlexNet_general_core, self).__init__()
        
        # Define features (Convolutional layers only)
        self.features = nn.Sequential(
            nn.Conv2d(channel_dict['features.0'][1], channel_dict['features.0'][0], kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(channel_dict['features.3'][1], channel_dict['features.3'][0], kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(channel_dict['features.6'][1], channel_dict['features.6'][0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(channel_dict['features.8'][1], channel_dict['features.8'][0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(channel_dict['features.10'][1], channel_dict['features.10'][0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        fc_in = channel_dict['features.10'][0] * 6 * 6

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 4096),  # Adapted `in_features`
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Linear(4096, 10),  # Output layer (10 classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

# Function to create the pruned AlexNet
def AlexNet_General(channel_dict, last_conv_shape):
    """
    - channel_dict: Dictionary with input and output channels for Conv2d layers.
    - last_conv_shape: Tuple containing (out_channels, height, width) of the last Conv2d layer.
    """
    # Calculate the input features for the first linear layer
    last_conv_out_features = last_conv_shape[0] * last_conv_shape[1] * last_conv_shape[2]
    return AlexNet_general(channel_dict, last_conv_out_features)

def AlexNet_General_core(channel_dict):
    """
    - channel_dict: Dictionary with input and output channels for Conv2d layers.
    - last_conv_shape: Tuple containing (out_channels, height, width) of the last Conv2d layer.
    """
    # Calculate the input features for the first linear layer
    return AlexNet_general_core(channel_dict)

def calculate_last_conv_out_features(pruned_model, input_size=(1, 3, 224, 224)):
    """
    Dynamically calculate the output features of the last convolutional layer after AdaptiveAvgPool2d.
    
    Parameters:
    - pruned_model: The pruned model with its layers.
    - input_size: The size of the input tensor (default: (1, 3, 224, 224)).
    
    Returns:
    - last_conv_out_features: The number of features for the first Linear layer.
    - last_conv_shape: Tuple containing (channels, height, width) of the last Conv2d layer.
    """
    pruned_model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        # Get the model's device
        device = next(pruned_model.parameters()).device
        
        # Move dummy input to the same device as the model
        dummy_input = torch.randn(input_size).to(device)
        
        # Pass the dummy input through the features and avgpool layers
        x = pruned_model.features(dummy_input)  # Pass through Conv2d layers
        x = pruned_model.avgpool(x)  # Pass through AdaptiveAvgPool2d
        
        last_conv_shape = x.shape[1:]  # (channels, height, width)
        last_conv_out_features = last_conv_shape[0] * last_conv_shape[1] * last_conv_shape[2]  # Flattened size
    
    return last_conv_out_features, last_conv_shape


def reconstruct_weights_from_dicts(model, pruned_indices, pruned_weights, unpruned_indices, unpruned_weights):
    """
    Reconstruct weights for a model using pruned and unpruned indices and tensors.

    Parameters:
    - pruned_indices: dict, mapping layer names to (dim0_indices, dim1_indices) of pruned weights.
    - pruned_weights: dict, mapping layer names to tensors of pruned weights.
    - unpruned_indices: dict, mapping layer names to (dim0_indices, dim1_indices) of unpruned weights.
    - unpruned_weights: dict, mapping layer names to tensors of unpruned weights.
    - model: torch.nn.Module, the model to reconstruct weights for.

    Returns:
    - reconstructed_model: torch.nn.Module, the model with reconstructed weights.
    """
    # Iterate through the model's state_dict to dynamically fetch layer shapes
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            if layer is None and layer_name.startswith("model."):
                layer_name = layer_name[len("model."):]
            new_device = layer.weight.device

            #Check if the layer is present or not
            if name not in pruned_indices or name not in unpruned_indices:
                print(f"Layer {name} not pruned, skipping weight reconstruction.")
                continue

            # Retrieve pruned and unpruned indices
            pruned_dim0, pruned_dim1 = pruned_indices[name].values()
            unpruned_dim0, unpruned_dim1 = unpruned_indices[name].values()

            # Skip if pruned_weights for this layer is empty
            if name not in pruned_weights or pruned_weights[name].numel() == 0:
                # No pruned weights to assign
                pass
            else:
                # Assign pruned weights
                if pruned_dim0 and pruned_dim1:
                    for i in range(len(pruned_dim0)):
                        out_idx = pruned_dim0[i]  # Output channel index
                        for j in range(len(pruned_dim1)):
                            in_idx = pruned_dim1[j]   # Input channel index
                            layer.weight.data[out_idx, in_idx, :, :] = pruned_weights[name][i, j].to(new_device)
                else:
                    for i in range(len(pruned_dim0)):
                        out_idx = pruned_dim0[i]  # Output channel index
                        layer.weight.data[out_idx, :, :, :] = pruned_weights[name][i].to(new_device)
                # Assign unpruned weights
                for i in range(len(unpruned_dim0)):
                        out_idx = unpruned_dim0[i]  # Output channel index
                        for j in range(len(unpruned_dim1)):
                            in_idx = unpruned_dim1[j]   # Input channel index
                            layer.weight.data[out_idx, in_idx, :, :] = unpruned_weights[name][i, j].to(new_device)

                # Channel Freezing
                for i in range(len(unpruned_dim0)):
                        out_idx = unpruned_dim0[i]  # Output channel index
                        for j in range(len(unpruned_dim1)):
                            in_idx = unpruned_dim1[j]   # Input channel index
                            layer.weight.data[out_idx, in_idx, :, :].requires_grad = False
                
    return model
                      
def reconstruct_Global_weights_from_dicts(model, pruned_indices, pruned_weights, unpruned_indices, unpruned_weights, freezing=False):
    """
    Reconstruct weights for a model using pruned and unpruned indices and tensors.

    Parameters:
    - pruned_indices: dict, mapping layer names to (dim0_indices, dim1_indices) of pruned weights.
    - pruned_weights: dict, mapping layer names to tensors of pruned weights.
    - unpruned_indices: dict, mapping layer names to (dim0_indices, dim1_indices) of unpruned weights.
    - unpruned_weights: dict, mapping layer names to tensors of unpruned weights.
    - model: torch.nn.Module, the model to reconstruct weights for.

    Returns:
    - reconstructed_model: torch.nn.Module, the model with reconstructed weights.
    """
    new_unpruned_dim0_freeze_list = {}
    new_unpruned_dim1_freeze_list = {}
    # Iterate through the model's state_dict to dynamically fetch layer shapes
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            new_device = layer.weight.device

            #Check if the layer is present or not
            if name not in pruned_indices or name not in unpruned_indices:
                print(f"Layer {name} not pruned, skipping weight reconstruction.")
                continue

            # Retrieve pruned and unpruned indices
            pruned_dim0, pruned_dim1 = pruned_indices[name].values()
            unpruned_dim0, unpruned_dim1 = unpruned_indices[name].values()
            
            # Combine and sort
            combined_dim0 = sorted(pruned_dim0 + unpruned_dim0)
            combined_dim1 = sorted(pruned_dim1 + unpruned_dim1)

            # Find new indices for list1 and list2
            new_pruned_dim0 = [combined_dim0.index(x) for x in pruned_dim0]
            new_unpruned_dim0 = [combined_dim0.index(x) for x in unpruned_dim0]

            new_pruned_dim1 = [combined_dim1.index(x) for x in pruned_dim1]
            new_unpruned_dim1 = [combined_dim1.index(x) for x in unpruned_dim1]

            new_unpruned_dim0_freeze_list[name] = new_unpruned_dim0
            new_unpruned_dim1_freeze_list[name] = new_unpruned_dim1

            # Assign pruned weights
            for i in range(len(new_pruned_dim0)):
                    out_idx = new_pruned_dim0[i]  # Output channel index
                    for j in range(len(new_pruned_dim1)):
                        in_idx = new_pruned_dim1[j]   # Input channel index
                        layer.weight.data[out_idx, in_idx, :, :] = pruned_weights[name][i, j].to(new_device)

            # Assign unpruned weights
            for i in range(len(new_unpruned_dim0)):
                    out_idx = new_unpruned_dim0[i]  # Output channel index
                    for j in range(len(new_unpruned_dim1)):
                        in_idx = new_unpruned_dim1[j]   # Input channel index
                        layer.weight.data[out_idx, in_idx, :, :] = unpruned_weights[name][i, j].to(new_device)

            # Channel Freezing --> NOT WORKING
            if freezing:
                for i in range(len(new_unpruned_dim0)):
                        out_idx = new_unpruned_dim0[i]  # Output channel index
                        for j in range(len(new_unpruned_dim1)):
                            in_idx = new_unpruned_dim1[j]   # Input channel index
                            layer.weight.data[out_idx, in_idx, :, :].requires_grad = False

                print(name, layer.weight.requires_grad)
                
    return model, new_unpruned_dim0_freeze_list, new_unpruned_dim1_freeze_list

# More efficient in use of memory
def zero_out_gradients_v2(model, dim0_indices, dim1_indices):
    for name, layer in model.named_modules():
        # if name not in dim0_indices or name not in dim1_indices:
        #         # print(f"Layer {name} not pruned, skipping weight reconstruction.")
        #         continue
        if isinstance(layer, nn.Conv2d) and layer.weight.grad is not None:
            dim0_idx = torch.tensor(dim0_indices[name], dtype=torch.long, device=layer.weight.grad.device)
            dim1_idx = torch.tensor(dim1_indices[name], dtype=torch.long, device=layer.weight.grad.device)

            # Ensure indices are within valid range
            dim0_idx = dim0_idx[dim0_idx < layer.weight.grad.shape[0]]
            dim1_idx = dim1_idx[dim1_idx < layer.weight.grad.shape[1]]

            # Create a grid of (dim0, dim1) combinations
            grid_dim0, grid_dim1 = torch.meshgrid(dim0_idx, dim1_idx, indexing='ij')

            # Zero out gradients at specified indices
            with torch.no_grad():
                layer.weight.grad[grid_dim0, grid_dim1, :, :] = 0

def freeze_channels(model, channel_dict): 
    """
    Freeze specific channels in convolutional layers based on the provided channel dictionary.

    Parameters:
    - model: The PyTorch model (e.g., ResNet-20).
    - channel_dict: A dictionary where keys are layer names, and values are dicts specifying
                    channels to freeze for dim 0 (output channels) and dim 1 (input channels).
                    Example:
                        {
                            "layer1.0.conv1": {"dim0": [0, 1], "dim1": [2, 3]},
                            "layer2.0.conv1": {"dim0": [4], "dim1": [1, 6]}
                        }

    Returns:
    - model: The modified model with specified channels frozen.
    """

    def freeze_grad_hook(grad, mask):
        grad_clone = grad.clone()
        grad_clone[mask] = 0  # Freeze specific channel weights
        return grad_clone

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) and name in channel_dict:
            freeze_info = channel_dict[name]
            dim0_channels = freeze_info.get("pruned_dim0", [])
            dim1_channels = freeze_info.get("pruned_dim1", [])
            
            weight = layer.weight
            # Create masks for the dimensions
            mask_dim0 = torch.zeros_like(weight, dtype=torch.bool)
            mask_dim0[dim0_channels, :, :, :] = True
            
            mask_dim1 = torch.zeros_like(weight, dtype=torch.bool)
            mask_dim1[:, dim1_channels, :, :] = True
                                
            # Combine masks to handle both dim 0 and dim 1
            combined_mask = mask_dim0 | mask_dim1

            weight.register_hook(lambda grad, mask=combined_mask: freeze_grad_hook(grad, mask))

    return model


def fine_tuner(model, train_loader, val_loader, device, pruning_percentage, fineTuningType, epochs, scheduler_type, patience=5, LR=1e-4):
    
    model = model.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Define LR scheduler
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, step_size_up=20, mode='triangular2')
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    lrs = []  # To track learning rates

    # Early stopping setup
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    # Fine-tuning loop
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = train_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Step the scheduler and track learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lrs.append(current_lr)
        
        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Accumulate validation metrics
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= total
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # **Log Metrics to Weights & Biases**
        wandb.log({
            f"{pruning_percentage}{scheduler_type}{fineTuningType}/Train Loss": epoch_loss,
            f"{pruning_percentage}{scheduler_type}{fineTuningType}/Train Accuracy": epoch_accuracy,
            f"{pruning_percentage}{scheduler_type}{fineTuningType}/Validation Loss": val_loss,
            f"{pruning_percentage}{scheduler_type}{fineTuningType}/Validation Accuracy": val_accuracy,
            f"{pruning_percentage}{scheduler_type}{fineTuningType}/Learning Rate": current_lr,
            "Epoch": epoch + 1
        })

        # Check for early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())  # Save best model
            patience_counter = 0  # Reset patience counter

            torch.save(model.state_dict(), f"checkpoint_{pruning_percentage}_{scheduler_type}_{fineTuningType}_{LR}_{epoch+1}.pth")

        else:
            patience_counter += 1  # Increment counter if no improvement
            
        print(f"Epoch [{epoch+1}/{epochs}], PruningPercentage: {pruning_percentage}, Scheduler: {scheduler_type}, "
          f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}, "
          f"Learning Rate: {current_lr:.6f}")
          
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered! Restoring best model from epoch {epoch - patience + 1}.")
            model.load_state_dict(best_model_wts)  # Restore best model weights
            break  # Stop training

    print("Fine-Tuning Complete")     
    return model  # Return the best model

def fine_tuner_zerograd(model, train_loader, val_loader, freeze_dim0, freeze_dim1, device, pruning_percentage, fineTuningType, epochs, scheduler_type, patience=25, LR=1e-4):
    
    model = model.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    
    # Define LR scheduler
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, step_size_up=20, mode='triangular2')
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    lrs = []  # To track learning rates

    # Early stopping setup
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    # Fine-tuning loop
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            zero_out_gradients_v2(model, freeze_dim0, freeze_dim1)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = train_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Step the scheduler and track learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lrs.append(current_lr)
        
        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Accumulate validation metrics
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= total
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # **Log Metrics to Weights & Biases**
        wandb.log({
            f"{pruning_percentage}{scheduler_type}{fineTuningType}/Train Loss": epoch_loss,
            f"{pruning_percentage}{scheduler_type}{fineTuningType}/Train Accuracy": epoch_accuracy,
            f"{pruning_percentage}{scheduler_type}{fineTuningType}/Validation Loss": val_loss,
            f"{pruning_percentage}{scheduler_type}{fineTuningType}/Validation Accuracy": val_accuracy,
            f"{pruning_percentage}{scheduler_type}{fineTuningType}/Learning Rate": current_lr,
            "Epoch": epoch + 1
        })

        # Check for early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())  # Save best model
            patience_counter = 0  # Reset patience counter

            torch.save(model.state_dict(), f"checkpoint_{pruning_percentage}_{scheduler_type}_{fineTuningType}_{LR}_{epoch+1}.pth")

        else:
            patience_counter += 1  # Increment counter if no improvement
            
        print(f"Epoch [{epoch+1}/{epochs}], PruningPercentage: {pruning_percentage}, Scheduler: {scheduler_type}, "
          f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}, "
          f"Learning Rate: {current_lr:.6f}")
          
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered! Restoring best model from epoch {epoch - patience + 1}.")
            model.load_state_dict(best_model_wts)  # Restore best model weights
            break  # Stop training

    print("Fine-Tuning Complete")     
    return model  # Return the best model


def copy_weights_from_dict(pruned_model, unpruned_weights):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data = unpruned_weights[name]
