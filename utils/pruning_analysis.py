import torch
import torch.nn as nn
from copy import deepcopy
from models.depGraph_fineTuner import DepGraphFineTuner
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR

import re

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

        print("MODEL AFTER PRUNING:\n", model)
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

    # Global storage to track pruned indices across ALL layers
    global_pruned_info = {layer_name: {'pruned_dim0': [], 'pruned_dim1': []} for layer_name in layers_to_prune.keys()}

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
                if target_layer_name in layers_to_prune:
                    print(f"🔍 Extracted Pruning: {pruning_type} on {target_layer_name}, Indices: {item.idxs}")

                    if pruning_type == "out":
                        global_pruned_info[target_layer_name]['pruned_dim0'].extend(item.idxs)
                    elif pruning_type == "in":
                        global_pruned_info[target_layer_name]['pruned_dim1'].extend(item.idxs)

    # Second pass: Apply pruning information to layers
    for layer_name, layer in module_dict.items():
        if layer_name not in layers_to_prune:  # Skip layers that aren't in layers_to_prune
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

    print("\n✅ Updated `num_pruned_channels`:", num_pruned_channels)
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



def get_core_weights(pruned_model, unpruned_weights):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            unpruned_weights[name] = module.weight.data

def fine_tuner(model, train_loader, val_loader, scheduler_type, device, num_epochs, LR=1e-3):

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
        scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=0.01, step_size_up=20, mode='triangular2')
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
        

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    lrs = []  # To track learning rates

    # Fine-tuning loop
    for epoch in range(num_epochs):
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}, Learning Rate: {current_lr:.6f}")
    
    del optimizer
           
    print("Fine-Tuning Complete")   

class ResNet_general(nn.Module):
    def __init__(self, block, num_blocks, channel_dict):
        super(ResNet_general, self).__init__()
        
        self.in_channels = channel_dict['conv1'][0]
        self.conv1 = nn.Conv2d(channel_dict['conv1'][1], channel_dict['conv1'][0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_dict['conv1'][0])
        self.layer1 = self._make_layer(block, channel_dict['layer1.0.conv1'][0], num_blocks[0])
        self.layer2 = self._make_layer(block, channel_dict['layer2.0.conv1'][0], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channel_dict['layer3.0.conv1'][0], num_blocks[2], stride=2)
        self.fc = nn.Linear(channel_dict['layer3.2.conv2'][0], 10)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out,(1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def Resnet_General(channel_dict):
        return ResNet_general(BasicBlock, [3,3,3], channel_dict)
            
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


    

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
            
            if name not in pruned_indices or name not in unpruned_indices:
                print(f"Layer {name} not found in pruned/unpruned indices. Skipping.")
                continue
            new_device = layer.weight.device

            # Retrieve pruned and unpruned indices
            pruned_dim0, pruned_dim1 = pruned_indices[name].values()
            unpruned_dim0, unpruned_dim1 = unpruned_indices[name].values()
            print(f"Layer: {name}")
            print(f"Expected shape: {layer.weight.shape}")
            print(f"Unpruned weights shape: {unpruned_weights[name].shape}")
            print(f"Unpruned dim0: {unpruned_dim0}, Unpruned dim1: {unpruned_dim1}")
            # Assign pruned weights
            layer.weight.data[pruned_dim0][:,pruned_dim1,:,:] = pruned_weights[name].to(new_device)

            # Assign unpruned weights
            # layer.weight.data[unpruned_dim0][:,unpruned_dim1,:,:] = unpruned_weights[name].to(new_device)
            if len(unpruned_dim0) == unpruned_weights[name].shape[0] and len(unpruned_dim1) == unpruned_weights[name].shape[1]:
                layer.weight.data[unpruned_dim0][:, unpruned_dim1, :, :] = unpruned_weights[name].to(new_device)
            else:
                print(f"Skipping layer {name} due to mismatched dimensions: "
                    f"Expected {[len(unpruned_dim0), len(unpruned_dim1)]}, but got {list(unpruned_weights[name].shape[:2])}")

            
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


import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F

class AlexNetLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
