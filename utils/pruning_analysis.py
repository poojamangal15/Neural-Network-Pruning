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


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")


# def get_pruned_info(groups, original_model):
#     """
#     Collect pruned indices and total counts for each layer in the model.

#     Parameters:
#     - groups: List of pruning groups obtained during pruning (from torch_pruning).
#     - original_model: The original, unpruned model (i.e., the one used to build the DepGraph).

#     Returns:
#     - pruned_info: Dict[layer_name] = {'pruned_dim0': set(), 'pruned_dim1': set()}
#          E.g. for Conv2d, dim0=out_channels, dim1=in_channels.
#          For Linear, dim0=out_features, dim1=in_features.
#     - num_pruned_channels: Dict[layer_name] = (num_pruned_out, num_pruned_in)
#     - pruned_weights: Dict[layer_name] = pruned weight tensor.
#     """
#     import torch
#     import torch.nn as nn

#     pruned_info = {}
#     num_pruned_channels = {}
#     pruned_weights = {}

#     # Build a lookup so we can reference layers by name
#     module_dict = dict(original_model.named_modules())

#     # Iterate over the groups
#     for layer_name, group in groups:
#         # Remove "model." prefix if present
#         if layer_name.startswith("model."):
#             short_name = layer_name[len("model."):]
#         else:
#             short_name = layer_name

#         layer = module_dict.get(short_name, None)
#         print("Layer", layer)
#         if layer is None:
#             print(f"[get_pruned_info] WARNING: '{layer_name}' not found. Skipping.")
#             continue

#         # If this layer isn't in pruned_info yet, initialize empty sets
#         if short_name not in pruned_info:
#             pruned_info[short_name] = {
#                 'pruned_dim0': set(),  # out_ch or out_features
#                 'pruned_dim1': set(),  # in_ch  or in_features
#             }

#         # Go through every (dep, idxs) inside this group's chain
#         for dep, idxs in group.items:
#             # `dep.target.module` is the actual module being pruned
#             tgt_module = dep.target.module
#             handler_str = str(dep.handler)

#             # Are we pruning out_channels or in_channels?
#             if "out_channels" in handler_str:
#                 # If it's a Conv2d, that means out_ch
#                 # If it's a Linear, that means out_features
#                 if tgt_module is layer:
#                     # Indices belong to this same "layer_name"
#                     pruned_info[short_name]['pruned_dim0'].update(idxs)

#             elif "in_channels" in handler_str:
#                 # Could be the conv in_ch or linear in_features
#                 if tgt_module is layer:
#                     pruned_info[short_name]['pruned_dim1'].update(idxs)

#         # After collecting sets, finalize them as sorted lists
#         dim0_list = sorted(pruned_info[short_name]['pruned_dim0'])
#         dim1_list = sorted(pruned_info[short_name]['pruned_dim1'])

#         # Count how many out/in channels got pruned
#         pruned_dim0_count = len(dim0_list)
#         pruned_dim1_count = len(dim1_list)
#         num_pruned_channels[short_name] = (pruned_dim0_count, pruned_dim1_count)

#         # Now gather pruned weights if it's a Conv2d or Linear
#         if isinstance(layer, nn.Conv2d):
#             if (pruned_dim0_count > 0 or pruned_dim1_count > 0):
#                 # e.g. out_ch pruned => gather weight[out_ch_pruned, ...]
#                 # in_ch pruned => gather weight[..., in_ch_pruned, ...]
#                 w = layer.weight.detach()
#                 pruned_weight = w[dim0_list][:, dim1_list, :, :]
#                 pruned_weights[short_name] = pruned_weight
#             else:
#                 pruned_weights[short_name] = torch.empty((0, 0))

#         elif isinstance(layer, nn.Linear):
#             if (pruned_dim0_count > 0 or pruned_dim1_count > 0):
#                 # For Linear, weight shape = [out_features, in_features]
#                 w = layer.weight.detach()
#                 pruned_weight = w[dim0_list][:, dim1_list]
#                 pruned_weights[short_name] = pruned_weight
#             else:
#                 pruned_weights[short_name] = torch.empty((0, 0))

#         else:
#             # Not a Conv2d or Linear, skip
#             pruned_weights[short_name] = torch.empty((0, 0))

#     # Debug print
#     print("\n[get_pruned_info] num_pruned_channels =>", num_pruned_channels)
#     return pruned_info, num_pruned_channels, pruned_weights

def prune_model(original_model, model, device, pruning_percentage=0.2, layer_pruning_percentages=None):
    pruned_info = {}
    
    model = model.to(device)
    example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

    # print("MODEL BEFORE PRUNING:\n", model.model)

    DG = tp.DependencyGraph().build_dependency(model.model, example_inputs)
    layers_to_prune = {
        "model.features.3": model.model.features[3],
        "model.features.6": model.model.features[6],
        "model.features.8": model.model.features[8],
        "model.features.10": model.model.features[10],
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
    pruned_info, num_pruned_channels, pruned_weights = get_pruned_info(groups, original_model)
    unpruned_info, num_unpruned_channels, unpruned_weights = get_unpruned_info(groups, original_model, pruned_info)

    pruned_and_unpruned_info = {"pruned_info": pruned_info, 
                                "num_pruned_channels": num_pruned_channels, 
                                "pruned_weights": pruned_weights, 
                                "unpruned_info": unpruned_info, 
                                "num_unpruned_channels": num_unpruned_channels, 
                                "unpruned_weights": unpruned_weights}
    return model, pruned_and_unpruned_info

def get_pruned_info(groups, original_model):
    """
    Collect pruned indices and total counts for each layer in the model.

    Parameters:
    - groups: List of pruning groups obtained during pruning (from torch_pruning).
    - original_model: The original, unpruned model (i.e., the one used to build the DepGraph).

    Returns:
    - pruned_info: A dictionary mapping layer_name -> {'pruned_dim0': [...], 'pruned_dim1': [...]}.
    - num_pruned_channels: A dictionary mapping layer_name -> (num_pruned_out, num_pruned_in).
    - pruned_weights: A dictionary mapping layer_name -> tensor of pruned weights for that layer.
    """

    pruned_info = {}
    num_pruned_channels = {}
    pruned_weights = {}

    module_dict = dict(original_model.named_modules())

    # --------------------------------------------------------------------------
    # GATHER PRUNED INDICES
    # --------------------------------------------------------------------------
    next_pruned_indices = []
    

    for layer_name, group in groups:
        for dep, pruned_idxs in group.items:
            mod = dep.target.module
            handler_str = str(dep.handler)

            # print("MOD HANDLER", mod, handler_str)

        layer = module_dict.get(layer_name, None)

        if layer is None and layer_name.startswith("model."):
            layer_name = layer_name[len("model."):]
            layer = module_dict.get(layer_name, None)

        if layer is None:
            print(f"Layer '{layer_name}' not found in original_model. Skipping.")
            continue

        # print(f"Processing layer: {layer_name}")

        # Initialize the sets for collecting pruned indices
        # used sets since they are a lot of repetitive values causing channel number mismatch
        if layer_name not in pruned_info:
            pruned_info[layer_name] = {
                'pruned_dim0': set(),  # out-ch
                'pruned_dim1': set()   # in-ch
            }

        if group.items:
            # Access the first Dependency object in group.items (others have the same indices)
            # print("group.items", group.items)
            first_item = group.items[0]
            last_item = group.items[-1]
            pruned_indices = first_item.idxs

            # print("last item", last_item)

            # Verify pruned indices against the layer's weight dimensions
            if pruned_indices:
                if max(pruned_indices) >= layer.weight.shape[0]:
                    print(f"Error: Pruned indices exceed dimensions for {layer_name}")
                    pruned_indices = [idx for idx in pruned_indices if idx < layer.weight.shape[0]]

            if "out_channels" in str(first_item):
                pruned_info[layer_name]['pruned_dim0'].update(pruned_indices)

                # Check for in_channels dependency in subsequent items
                for subsequent_item in group.items[1:]:
                    if "in_channels" in str(subsequent_item):
                        # if "Linear" in str(subsequent_item):
                        #     pruned_info["classifier.1"] = {
                        #         'pruned_dim0': set(),  # out-ch
                        #         'pruned_dim1': set()   # in-ch
                        #     }
                        #     pruned_info["classifier.1"]['pruned_dim1'].update(last_item.idxs)
                        # print("in for loop in channels", next_pruned_indices)
                        pruned_info[layer_name]['pruned_dim1'].update(next_pruned_indices)
                        next_pruned_indices = subsequent_item.idxs
                        break  # No need to check further, as you've found the in_channels dependency

            elif "in_channels" in str(first_item):
                # print("in channels no for")
                pruned_info[layer_name]['pruned_dim1'].update(pruned_indices)

    # --------------------------------------------------------------------------
    # FINALIZE pruned_info, num_pruned_channels, and pruned_weights
    # --------------------------------------------------------------------------
    for layer_name in pruned_info:

        dim0_list = sorted(pruned_info[layer_name]['pruned_dim0'])
        dim1_list = sorted(pruned_info[layer_name]['pruned_dim1'])

        pruned_info[layer_name]['pruned_dim0'] = dim0_list
        pruned_info[layer_name]['pruned_dim1'] = dim1_list

        pruned_dim0_count = len(dim0_list)
        pruned_dim1_count = len(dim1_list)
        num_pruned_channels[layer_name] = (pruned_dim0_count, pruned_dim1_count)

        layer = module_dict.get(layer_name, None)
        if layer is None:
            print(f"Layer {layer_name} not found in model. Skipping pruned_weights.")
            pruned_weights[layer_name] = torch.empty((0, 0))
            continue

        if isinstance(layer, nn.Conv2d) and (pruned_dim0_count > 0 or pruned_dim1_count > 0):
            weights = layer.weight.detach()
            # pruned_weight_tensor = weights[dim0_list][:, dim1_list, :, :]
            if dim0_list and dim1_list:
                pruned_weights[layer_name] = weights[dim0_list, :, :, :][:, dim1_list, :, :]
            else:
                pruned_weights[layer_name] = weights[dim0_list, :, :, :]
            # pruned_weights[layer_name] = pruned_weight_tensor
        else:
            pruned_weights[layer_name] = torch.empty((0, 0))

    # print("\n[DEBUG] pruned Info summary:")
    # for ln, info in pruned_info.items():
    #     print(f"  -> {ln}, pruned_dim0={info['pruned_dim0']}, unpruned_dim1={info['pruned_dim1']}")

    print("\nnum_pruned_channels:", num_pruned_channels)
    print("features.3 weights-------------", pruned_weights)
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

    print("\nnum pruned channels:", num_pruned_channels)
    print("\nnum_unpruned_channels:", num_unpruned_channels)

    return unpruned_info, num_unpruned_channels, unpruned_weights



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

def get_rebuild_channels(unpruned_channels, pruned_channels):
    new_channels_dict = {}
    for name, weight in pruned_channels.items():
        new_in_channels = int(unpruned_channels[name][1] + pruned_channels[name][1])
        new_out_channels = int(unpruned_channels[name][0] + pruned_channels[name][0])
        
        new_channels_dict[name] = (new_in_channels, new_out_channels)
        # print("newchannels deictionary", new_channels_dict)
    
    if "features.0" not in new_channels_dict:
    # Assuming original AlexNet has 64 output channels and 3 input channels in the first layer
        new_channels_dict["features.0"] = (64, 3)
    return new_channels_dict

def get_core_weights(pruned_model, unpruned_weights):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
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

    def fine_tune_model(self, train_dataloader, val_dataloader, device, epochs=1, learning_rate=1e-5):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        self.to(device).to(torch.float32)  # ensure model is float32 on the chosen device

        for epoch in range(epochs):
            self.train()
            for batch in train_dataloader:
                inputs, targets = batch
                
                # Move inputs and targets to the same device and dtype
                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            self.eval()
            with torch.no_grad():
                total_loss = 0
                correct = 0
                for batch in val_dataloader:
                    inputs, targets = batch
                    
                    # Again, move inputs/targets to correct device & dtype
                    inputs = inputs.to(device, dtype=torch.float32)
                    targets = targets.to(device)
                    
                    outputs = self(inputs)
                    total_loss += criterion(outputs, targets).item()
                    correct += (outputs.argmax(dim=1) == targets).sum().item()
                
                val_accuracy = correct / len(val_dataloader.dataset)
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Validation Accuracy: {val_accuracy:.4f}, "
                    f"Loss: {total_loss:.4f}"
                )

# Function to create the pruned AlexNet
def AlexNet_General(channel_dict, last_conv_shape):
    """
    - channel_dict: Dictionary with input and output channels for Conv2d layers.
    - last_conv_shape: Tuple containing (out_channels, height, width) of the last Conv2d layer.
    """
    # Calculate the input features for the first linear layer
    last_conv_out_features = last_conv_shape[0] * last_conv_shape[1] * last_conv_shape[2]
    return AlexNet_general(channel_dict, last_conv_out_features)

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
            
            new_device = layer.weight.device

            #Check if the layer is present or not
            if name not in pruned_indices or name not in unpruned_indices:
                print(f"Layer {name} not pruned, skipping weight reconstruction.")
                continue

            # Retrieve pruned and unpruned indices
            pruned_dim0, pruned_dim1 = pruned_indices[name].values()
            unpruned_dim0, unpruned_dim1 = unpruned_indices[name].values()

            print("pruned weights-----------------", pruned_weights)
            # Skip if pruned_weights for this layer is empty
            if name not in pruned_weights or pruned_weights[name].numel() == 0:
                print("Passing layer--------", name)
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
                
                print(f"Layer {name}: Rebuilt weight shape = {layer.weight.data.shape}")
                print(f"Unpruned indices (output): {unpruned_dim0}")
                print(f"Unpruned indices (input): {unpruned_dim1}")
                print(f"Stored unpruned weights shape: {unpruned_weights[name].shape}")

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


def fine_tuner(model, train_loader, val_loader, device, epochs, scheduler_type, patience=3, LR=1e-4):
    
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
            f"{scheduler_type}/Train Loss": epoch_loss,
            f"{scheduler_type}/Train Accuracy": epoch_accuracy,
            f"{scheduler_type}/Validation Loss": val_loss,
            f"{scheduler_type}/Validation Accuracy": val_accuracy,
            f"{scheduler_type}/Learning Rate": current_lr,
            "Epoch": epoch + 1
        })

        # Check for early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())  # Save best model
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment counter if no improvement
            
        print(f"Epoch [{epoch+1}/{epochs}], Scheduler: {scheduler_type}, "
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
