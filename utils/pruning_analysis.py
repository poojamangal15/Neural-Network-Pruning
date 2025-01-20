import torch
import torch.nn as nn
from copy import deepcopy
from models.depGraph_fineTuner import DepGraphFineTuner
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")
                        
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


def Resnet_General(channel_dict):
    return ResNet_general(BasicBlock, [3,3,3], channel_dict)

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

            # Assign pruned weights
            layer.weight.data[pruned_dim0][:,pruned_dim1,:,:] = pruned_weights[name].to(new_device)

            # Assign unpruned weights
            layer.weight.data[unpruned_dim0][:,unpruned_dim1,:,:] = unpruned_weights[name].to(new_device)
            
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
