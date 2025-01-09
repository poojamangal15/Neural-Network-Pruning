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

        weights = layer.weight.detach()
        if pruned_dim0 or pruned_dim1:  # If there are pruned dimensions
            pruned_weight_tensor = weights[pruned_dim0][:, pruned_dim1, :, :]
            pruned_weights[layer_name] = pruned_weight_tensor

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
            if name == "features.0":
                new_in_channel = 3  # Input to the first Conv2d is always 3 (RGB images)
                new_out_channel = module.weight.data.shape[0] + pruned_dict.get(name, (0, 0))[0]
            else:
                new_in_channel = module.weight.data.shape[1] + pruned_dict.get(name, (0, 0))[1]
                new_out_channel = module.weight.data.shape[0] + pruned_dict.get(name, (0, 0))[0]

            new_channel_dict[name] = (int(new_out_channel), int(new_in_channel))

    return new_channel_dict



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