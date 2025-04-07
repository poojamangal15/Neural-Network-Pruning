import torch
import torch.nn as nn
from copy import deepcopy
import copy
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR
import torch_pruning as tp
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from tqdm import tqdm
import re
import torch.nn.functional as F

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")

def prune_model(original_model, model, device, pruning_percentage=0.2):
    pruned_info = {}
    
    model = model.to(device)
    example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

    DG = tp.DependencyGraph().build_dependency(model, example_inputs)
    layers_to_prune = {
        "layer1.0.conv1": model.layer1[0].conv1,
        "layer1.0.conv2": model.layer1[0].conv2,
        "layer1.1.conv1": model.layer1[1].conv1,
        "layer1.1.conv2": model.layer1[1].conv2,
        "layer1.2.conv1": model.layer1[2].conv1,
        "layer1.2.conv2": model.layer1[2].conv2,
        "layer2.0.conv1": model.layer2[0].conv1,
        "layer2.0.conv2": model.layer2[0].conv2,
        "layer2.1.conv1": model.layer2[1].conv1,
        "layer2.1.conv2": model.layer2[1].conv2,
        "layer2.2.conv1": model.layer2[2].conv1,
        "layer2.2.conv2": model.layer2[2].conv2,
        "layer3.0.conv1": model.layer3[0].conv1,
        "layer3.0.conv2": model.layer3[0].conv2,
        "layer3.1.conv1": model.layer3[1].conv1,
        "layer3.1.conv2": model.layer3[1].conv2,
        "layer3.2.conv1": model.layer3[2].conv1,
        "layer3.2.conv2": model.layer3[2].conv2,
    }



    def get_pruning_indices(module, percentage):
        with torch.no_grad():
            weight = module.weight.data
            if isinstance(module, torch.nn.Conv2d):
                channel_norms = weight.abs().mean(dim=[1,2,3])  
            else:
                return None

            pruning_count = int(channel_norms.size(0) * percentage)
            # Ensure at least 1 channel remains
            max_prune = channel_norms.size(0) - 1  
            pruning_count = min(pruning_count, max_prune)  # Never prune all channels

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
        print(f"Pruning with {pruning_percentage*100}% percentage on {len(groups)} layers...")
        for layer_name, group in groups:
            print(f"Pruning layer: {layer_name}")
            group.prune()

        # print("MODEL AFTER PRUNING:\n", model)
    else:
        print("No valid pruning groups found. The model was not pruned.")

    # Check for all the pruned and unpruned indices and weights    
    pruned_info, num_pruned_channels, pruned_weights = get_pruned_info(groups, original_model, layers_to_prune)
    unpruned_info, num_unpruned_channels, unpruned_weights = get_unpruned_info(groups, original_model, pruned_info)

    pruned_and_unpruned_info = {"pruned_info": pruned_info, 
                                "num_pruned_channels": num_pruned_channels, 
                                "pruned_weights": pruned_weights, 
                                "unpruned_info": unpruned_info, 
                                "num_unpruned_channels": num_unpruned_channels, 
                                "unpruned_weights": unpruned_weights}
    return model, pruned_and_unpruned_info

def high_level_pruner(original_model, model, device, pruning_percentage=0.2, layer_pruning_percentages=None):
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
    # ignored_layers.append(model.conv1)

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
        # print(group)
            # print("Layer name", layer_name)
            # print("Pruned info of layer", pruned_info[layer_name])
        group.prune()    

    # print("num pruned info", num_pruned_channels)
    unpruned_info, num_unpruned_channels, unpruned_weights = get_unpruned_info_high_level(original_model, pruned_info)

    pruned_and_unpruned_info = {"pruned_info": pruned_info, 
                                "num_pruned_channels": num_pruned_channels, 
                                "pruned_weights": pruned_weights, 
                                "unpruned_info": unpruned_info, 
                                "num_unpruned_channels": num_unpruned_channels, 
                                "unpruned_weights": unpruned_weights}
    
    return model, pruned_and_unpruned_info

def high_level_prunerTaylor(original_model, model, device, pruning_percentage=0.2, layer_pruning_percentages=None):
    """
    Prunes the model using Torch-Pruning's Taylor Importance Pruner.

    Parameters:
        - model (nn.Module): The PyTorch model to be pruned.
        - example_inputs (torch.Tensor): Sample input for pruning.
        - pruning_ratio (float): Percentage of channels to prune.
        - iterative_steps (int): Number of iterative pruning steps.

    Returns:
        - pruned_model (nn.Module): The pruned PyTorch model.
        - pruned_info (dict): Dictionary containing pruned indices.
    """
    model.train()  # Ensure model is in training mode for gradients
    
    example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

    # ✅ Switch to Taylor-based Importance (requires gradients)
    imp = tp.importance.TaylorImportance()

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)
    # ignored_layers.append(model.conv1)

    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,  # Taylor-based pruning
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_percentage,
        ignored_layers=ignored_layers,
    )

    # ✅ **Step 1: Compute Gradients Before Pruning**
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Use any optimizer
    criterion = nn.CrossEntropyLoss()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)
    dummy_target = torch.tensor([0], dtype=torch.long, device=device)  # Fake target

    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()  # Compute gradients before pruning

    # Dictionary to store formatted pruning info
    pruned_weights = {}
    num_pruned_channels = {}
    pruned_info = {}

    for group in pruner.step(interactive=True):  # Now gradients exist, Taylor importance works!
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
                layer_name = layer_name[len("model."):]

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

        group.prune()

    # Get unpruned information
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

def hessian_based_pruner(original_model, model, device, train_loader, pruning_percentage=0.2):
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
    imp = tp.importance.HessianImportance()  # Use default settings (optionally add normalize or other args)

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)
    # Optionally, you may also ignore the first conv layer if desired:
    # ignored_layers.append(model.conv1)

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

    # ignored_layers.append(model.conv1)

    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        global_pruning=True,
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_percentage, 
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
        # print(model)
        # print(model(example_inputs).shape)
        # print(
        #     "  Iter %d/%d, Params: %.2f M => %.2f M"
        #     % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
        # )
        # print(
        #     "  Iter %d/%d, MACs: %.2f G => %.2f G"
        #     % (i+1, iterative_steps, base_macs / 1e9, macs / 1e9)
        # )
        # count_zero_weights(model)
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


def get_unpruned_info_high_level(model, pruned_info):
    """
    Derives which indices were not pruned (the complement of pruned indices).
    Returns unpruned_info, a dict for each layer with "unpruned_dim0" and "unpruned_dim1";
    num_unpruned_channels, and unpruned_weights.
    """

    unpruned_info = {}
    num_unpruned_channels = {}
    unpruned_weights = {}

    # Utility to build sorted list of unpruned indices
    def get_unpruned_indices(total_count, pruned_list):
        """Return sorted list of indices not in pruned_list."""
        all_indices = set(range(total_count))
        pruned_set = set(pruned_list)
        unpruned_set = all_indices - pruned_set
        return sorted(list(unpruned_set))

    for name, module in model.named_modules():
        # If the module wasn't pruned at all in your workflow, you can optionally store "no prunes" or skip
        if name not in pruned_info:
            # By default, assume no pruning took place on this layer
            if isinstance(module, nn.Conv2d):
                out_channels, in_channels = module.weight.data.shape[:2]
                unpruned_dim0 = list(range(out_channels))
                unpruned_dim1 = list(range(in_channels))
            elif isinstance(module, nn.Linear):
                out_features, in_features = module.weight.data.shape
                unpruned_dim0 = list(range(out_features))
                unpruned_dim1 = list(range(in_features))
            else:
                # If it doesn't match conv/linear, skip or store empty
                continue

            unpruned_info[name] = {
                "unpruned_dim0": unpruned_dim0,
                "unpruned_dim1": unpruned_dim1
            }
            num_unpruned_channels[name] = (len(unpruned_dim0), len(unpruned_dim1))

            # If you want to store the entire (unpruned) weight, you can do so:
            unpruned_weights[name] = module.weight.data.clone()

        else:
            pruned_dim0 = pruned_info[name].get("pruned_dim0", [])
            pruned_dim1 = pruned_info[name].get("pruned_dim1", [])

            if isinstance(module, nn.Conv2d):
                # Weight shape: (out_channels, in_channels, kH, kW)
                out_channels, in_channels = module.weight.data.shape[:2]
                unpruned_dim0 = get_unpruned_indices(out_channels, pruned_dim0)
                unpruned_dim1 = get_unpruned_indices(in_channels, pruned_dim1)

                # Slice out those unpruned channels
                w = module.weight.data  # shape [outC, inC, kH, kW]
                unpruned_w = w[unpruned_dim0][:, unpruned_dim1, :, :].clone()

            elif isinstance(module, nn.Linear):
                # Weight shape: (out_features, in_features)
                out_features, in_features = module.weight.data.shape
                unpruned_dim0 = get_unpruned_indices(out_features, pruned_dim0)
                unpruned_dim1 = get_unpruned_indices(in_features, pruned_dim1)

                # Slice out those unpruned channels
                w = module.weight.data  # shape [outF, inF]
                unpruned_w = w[unpruned_dim0][:, unpruned_dim1].clone()

            else:
                # If not Conv2d or Linear, skip or handle differently
                continue

            # Store results
            unpruned_info[name] = {
                "unpruned_dim0": unpruned_dim0,
                "unpruned_dim1": unpruned_dim1
            }
            num_unpruned_channels[name] = (len(unpruned_dim0), len(unpruned_dim1))
            unpruned_weights[name] = unpruned_w

    # print("num unpruned channels", num_unpruned_channels)
    return unpruned_info, num_unpruned_channels, unpruned_weights


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

    all_conv2d_channels = ["layer1.0.conv1", "layer1.0.conv2", "layer1.1.conv1", "layer1.1.conv2", "layer1.2.conv1", "layer1.2.conv2", "layer2.0.conv1", "layer2.0.conv2", "layer2.1.conv1", "layer2.1.conv2", "layer2.2.conv1", "layer2.2.conv2", "layer3.0.conv1", "layer3.0.conv2", "layer3.1.conv1", "layer3.1.conv2", "layer3.2.conv1","layer3.2.conv2"]
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
    import torch
    
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

        # If it's Conv2d, store shape info
        if isinstance(layer, torch.nn.Conv2d):
            total_output_channels = layer.weight.shape[0]
            total_input_channels  = layer.weight.shape[1]
            # print(f"      [DEBUG] out_ch={total_output_channels}, in_ch={total_input_channels}")

        # Build all_output_indices and all_input_indices
        all_output_indices = list(range(total_output_channels)) if total_output_channels else []
        all_input_indices  = list(range(total_input_channels))  if total_input_channels else []

        pruned_dim0 = pruned_info[layer_name]['pruned_dim0']
        pruned_dim1 = pruned_info[layer_name]['pruned_dim1']

        # print("pruned dim0", pruned_dim0)
        # print("pruned dim1", pruned_dim1)

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

    # print("\nnum pruned channels:", num_pruned_channels)
    # print("\nnum_unpruned_channels:", num_unpruned_channels)

    return unpruned_info, num_unpruned_channels, unpruned_weights

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
    # print("num pruned channels", num_pruned_channels)
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
    # print("num unpruned channels", num_unpruned_channels)

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

def extend_channels(model, pruned_dict):
    """
    Extend channel dictionary to include all layers, especially the first layer.
    """
    new_channel_dict = {}

    for name, module in model.named_modules():
        if name.startswith("model."):
            name = name[len("model."):]

        if isinstance(module, nn.Conv2d):
            # Handle the first layer (features.0) differently
            # print("name, pruned dict", name, pruned_dict.get(name, (0, 0))[0])
            # print("module.weight" ,module.weight.data.shape[1])

            new_in_channel = module.weight.data.shape[1] + pruned_dict.get(name, (0, 0))[1]
            new_out_channel = module.weight.data.shape[0] + pruned_dict.get(name, (0, 0))[0]
            # print("new in and out channels", new_in_channel, new_out_channel)

            new_channel_dict[name] = (int(new_out_channel), int(new_in_channel))

    return new_channel_dict



def get_core_weights(pruned_model, unpruned_weights):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            unpruned_weights[name] = module.weight.data


# class ResNet_general(nn.Module):
#     def __init__(self, block, num_blocks, channel_dict):
#         super(ResNet_general, self).__init__()
        
#         print("CHANNEL DICT", channel_dict)
#         self.in_channels = channel_dict['conv1'][0]
#         self.conv1 = nn.Conv2d(channel_dict['conv1'][1], channel_dict['conv1'][0], kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(channel_dict['conv1'][0])
#         self.layer1 = self._make_layer(block, channel_dict['layer1.0.conv1'][0], num_blocks[0])
#         self.layer2 = self._make_layer(block, channel_dict['layer2.0.conv1'][0], num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, channel_dict['layer3.0.conv1'][0], num_blocks[2], stride=2)
#         self.fc = nn.Linear(channel_dict['layer3.2.conv2'][0], 10)

#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         layers = []
#         layers.append(block(self.in_channels, out_channels))
#         self.in_channels = out_channels
#         for _ in range(1, blocks):
#             layers.append(block(out_channels, out_channels))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.adaptive_avg_pool2d(out,(1, 1))
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out

# def Resnet_General(channel_dict):
#         return ResNet_general(BasicBlock, [3,3,3], channel_dict)
            
# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.downsample = nn.Sequential()
#         if in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.downsample(x)
#         out = F.relu(out)
#         return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # Apply stride to conv1 (to reduce spatial size if stride=2)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # Conv2 always has stride=1 (does not change spatial resolution)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample path (only applies when stride=2 or channels change)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)  # Apply downsample to residual path
        out = F.relu(self.bn1(self.conv1(x)))  # Apply stride in conv1 if needed
        out = self.bn2(self.conv2(out))  # Keep conv2 stride=1
        out += residual  # Make sure shapes match
        return F.relu(out)

    
class ResNet_general(nn.Module):
    def __init__(self, block, num_blocks, channel_dict):
        super(ResNet_general, self).__init__()

        self.in_channels = channel_dict['conv1'][0]
        self.conv1 = nn.Conv2d(channel_dict['conv1'][1], channel_dict['conv1'][0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_dict['conv1'][0])
        
        # Pass `stride=2` for layer2 and layer3 (since they reduce resolution)
        self.layer1 = self._make_layer(block, num_blocks[0], channel_dict, prefix='layer1', stride=1)
        self.layer2 = self._make_layer(block, num_blocks[1], channel_dict, prefix='layer2', stride=2)
        self.layer3 = self._make_layer(block, num_blocks[2], channel_dict, prefix='layer3', stride=2)

        self.fc = nn.Linear(channel_dict['layer3.2.conv2'][0], 10)
    
    def _make_layer(self, block, num_blocks, channel_dict, prefix, stride=1):
        layers = []
        
        for i in range(num_blocks):
            conv1_key = f'{prefix}.{i}.conv1'
            conv2_key = f'{prefix}.{i}.conv2'
            downsample_key = f'{prefix}.{i}.downsample.0'  # Only if needed

            in_channels = channel_dict[conv1_key][1]
            mid_channels = channel_dict[conv1_key][0]
            out_channels = channel_dict[conv2_key][0]

            # Apply stride=2 for the first block in the layer (to reduce spatial size)
            current_stride = stride if i == 0 else 1

            layers.append(block(in_channels, mid_channels, out_channels, stride=current_stride))
            
            # Ensure in_channels is correctly updated for next block
            in_channels = out_channels

        return nn.Sequential(*layers)


    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def Resnet_General(channel_dict):
    return ResNet_general(BasicBlock, [3, 3, 3], channel_dict)


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
            if layer is None and name.startswith("model."):
                name = name[len("model."):]
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
                        print(f"{name}: Assigning weight[{out_idx}, {in_idx}] shape {unpruned_weights[name][i, j].shape} -> layer.weight.data[{out_idx}, {in_idx}].shape = {layer.weight.data[out_idx, in_idx].shape}")
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


def debug_pruning_info(
    original_model: torch.nn.Module,
    pruned_model: torch.nn.Module,
    pruned_dict: dict,
    unpruned_dict: dict
):
    """
    Prints out channel dimensions for Conv2d layers in both the original 
    and pruned model, along with the values from `pruned_dict`.
    
    - `pruned_dict` is typically a dict like:
          {
              "features.3": (pruned_out, pruned_in),
              "features.6": (pruned_out, pruned_in),
              ...
          }
      where each tuple indicates how many out-channels and in-channels 
      were pruned for that layer.
    """
    print("========== ORIGINAL MODEL CHANNELS ==========")
    for name, module in original_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"[{name}] out_ch = {module.out_channels}, in_ch = {module.in_channels}")
    
    print("\n========== PRUNED MODEL CHANNELS ==========")
    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"[{name}] out_ch = {module.out_channels}, in_ch = {module.in_channels}")
    
    print("\n========== PRUNED_DICT CONTENTS ==========")
    for name, val in pruned_dict.items():
        # val might be (pruned_out, pruned_in) or something similar
        # so let's assume that's the format
        print(f"[{name}] -> pruned_out = {val[0]}, pruned_in = {val[1]}")

    print("\n========== PRUNED_DICT ==========")
    for name, val in pruned_dict.items():
        # e.g., val might be (pruned_out, pruned_in) or a dict with indices
        print(f"[{name}] -> {val}")

    print("\n========== UNPRUNED_DICT ==========")
    for name, val in unpruned_dict.items():
        print(f"[{name}] -> {val}")

def copy_weights_from_dict(pruned_model, unpruned_weights):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) and name in unpruned_weights.keys():
            module.weight.data = unpruned_weights[name]

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