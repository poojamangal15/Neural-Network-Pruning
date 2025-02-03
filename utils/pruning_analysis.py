import torch
import torch.nn as nn
import os
from copy import deepcopy
from models.depGraph_fineTuner import DepGraphFineTuner
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    CyclicLR,
    CosineAnnealingLR,
    ReduceLROnPlateau
)
                     
def count_parameters(model):
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_in_mb(model):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / (1024 * 1024)
    os.remove("temp.p")
    return size_mb

def get_pruned_info(groups, original_model):
    """
    Collect pruned indices and total counts for each layer in the model.

    Parameters:
    - groups: List of pruning groups obtained during pruning (from torch_pruning or similar).
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
            # Access the first Dependency object in group.items
            first_item = group.items[0]
            pruned_indices = first_item.idxs

            # Verify pruned indices against the layer's weight dimensions
            if pruned_indices:
                if max(pruned_indices) >= layer.weight.shape[0]:
                    print(f"Error: Pruned indices exceed dimensions for {layer_name}")
                    pruned_indices = [idx for idx in pruned_indices if idx < layer.weight.shape[0]]

            # print("First item", first_item)
            if "out_channels" in str(first_item):
                pruned_info[layer_name]['pruned_dim0'].update(pruned_indices)

                # Check for in_channels dependency in subsequent items
                for subsequent_item in group.items[1:]:
                    if "in_channels" in str(subsequent_item):
                        pruned_info[layer_name]['pruned_dim1'].update(next_pruned_indices)
                        next_pruned_indices = subsequent_item.idxs
                        break  # No need to check further, as you've found the in_channels dependency

            elif "in_channels" in str(first_item):
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
            pruned_weight_tensor = weights[dim0_list][:, dim1_list, :, :]
            pruned_weights[layer_name] = pruned_weight_tensor
        else:
            pruned_weights[layer_name] = torch.empty((0, 0))

    # print("\n[DEBUG] pruned Info summary:")
    # for ln, info in pruned_info.items():
    #     print(f"  -> {ln}, pruned_dim0={info['pruned_dim0']}, unpruned_dim1={info['pruned_dim1']}")

    print("\nnum_pruned_channels:", num_pruned_channels)
    return pruned_info, num_pruned_channels, pruned_weights


def get_unpruned_info(groups, original_model):
    import torch
    
    unpruned_info = {}
    num_unpruned_channels = {}
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

        # Track pruned indices
        pruned_dim0 = []
        pruned_dim1 = []
        
        # print("  [DEBUG] group items:")
        if group.items:
            # Access the first Dependency object in group.items
            first_item = group.items[0]

            # Get the indices from the first Dependency object
            pruned_indices = first_item.idxs

            # print(f"First occurrence of indices: {pruned_indices}")

            # Check if these are out_channels or in_channels
            if "out_channels" in str(first_item):
                pruned_dim0.extend(pruned_indices)
            elif "in_channels" in str(first_item):
                pruned_dim1.extend(pruned_indices)

                
        # Deduplicate in case the same indices appear multiple times
        pruned_dim0 = sorted(set(pruned_dim0))
        pruned_dim1 = sorted(set(pruned_dim1))

        # Compute the unpruned
        unpruned_dim0 = [i for i in all_output_indices if i not in pruned_dim0]
        unpruned_dim1 = [i for i in all_input_indices  if i not in pruned_dim1]
        
        # print(f"  [DEBUG] pruned_dim0={pruned_dim0}")
        # print(f"  [DEBUG] pruned_dim1={pruned_dim1}")
        # print(f"  [DEBUG] unpruned_dim0={unpruned_dim0}")
        # print(f"  [DEBUG] unpruned_dim1={unpruned_dim1}")
        
        unpruned_info[layer_name]['unpruned_dim0'] = unpruned_dim0
        unpruned_info[layer_name]['unpruned_dim1'] = unpruned_dim1
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

    # Final debug summary
    # print("\n[DEBUG] Unpruned Info summary:")
    # for ln, info in unpruned_info.items():
    #     print(f"  -> {ln}, unpruned_dim0={info['unpruned_dim0']}, unpruned_dim1={info['unpruned_dim1']}")
    # print("\n[DEBUG] num_unpruned_channels:", num_unpruned_channels)
    
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
    # print("reconstruct_weights_from_dicts---------", model)
                
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

def fine_tuner(model, train_loader, val_loader, num_epochs, optim_type, scheduler_type, exp_name, device, LR=1e-3):

    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    if optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Define LR scheduler
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=0.01, step_size_up=20, mode='triangular2')
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        

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

        # Step the scheduler and track learning rate
        if scheduler_type == 'Default':
            scheduler.step(val_loss)
        else:    
            scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        lrs.append(current_lr)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}, Learning Rate: {current_lr:.6f}")
    
           
    print("Fine-Tuning Complete")          