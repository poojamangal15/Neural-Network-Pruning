import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle
import numpy as np
from ptflops import get_model_complexity_info
from calflops import calculate_flops
from thop import profile
from deepspeed.profiling.flops_profiler import get_model_profile


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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, 10)

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

c


#THIS FUNCTION IS USED FOR DENSE MODEL CREATION ----------------------->
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

#THIS FUNCTION IS USED FOR DENSE MODEL CREATION ----------------------->
def Resnet_General(channel_dict):
    return ResNet_general(BasicBlock, [3,3,3], channel_dict)


#LOOKS LIKE IT is used for creating dense model only
def rebuild_resnet(model, p, first_channel=16):
    new_model = ResNet_prune(BasicBlock, [3,3,3], prune_ratio=p, first_channel=first_channel)
    
    # Create a list to hold the layers and corresponding weights
    layer_weights = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Get the mask used for pruning
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask.bool()  # The mask indicates the remaining channels
                remaining_weights = module.weight.data[mask].clone()
                out_channels, in_channels, kernel_height, kernel_width = module.weight.shape
                remaining_weights = remaining_weights.view(-1, in_channels, kernel_height, kernel_width)
                
                # print(name, module.weight.data.shape, mask.shape, remaining_weights.shape)
                
                # If it's the first layer, directly assign to new model's first conv layer
                if name == 'conv1':
                    new_model.conv1.weight.data = remaining_weights
                else:
                    # Update other layers based on their names
                    layer_weights.append((name, remaining_weights))

    # Now we have the remaining weights for each layer, let's set them
    for name, weights in layer_weights:
        layer_name = name.replace('.', '_')  # Change '.' to '_' for attribute access
        new_model_state_dict = new_model.state_dict()
        new_model_state_dict[layer_name + '.weight'] = weights
        new_model.load_state_dict(new_model_state_dict, strict=False)

    return new_model


#THIS FUNCTION IS USED FOR DENSE MODEL CREATION -----------------------> 
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

#THIS FUNCTION IS USED FOR DENSE MODEL CREATION ----------------------->
# Function to copy weights from pruned model to reduced model
def copy_weights(pruned_model, rebuilt_model, nonzero_channels):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            n=0
            pruned_layer = module
            reduced_layer = dict(rebuilt_model.named_modules())[name]
            
            out_channels, in_channels,_,_ = reduced_layer.weight.shape
        
            with torch.no_grad():
                for i in range(out_channels):
                    for j in range(in_channels):
                        dim0, dim1 = nonzero_channels[name][n]
                        reduced_layer.weight.data[i,j] = pruned_layer.weight.data[dim0, dim1]
                        n += 1
     
     
                
def copy_weights_from_dict(pruned_model, unpruned_weights):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data = unpruned_weights[name]


# def rebuilt_bigger(channel_dict_path, state_dict_path, device):
    
#     with open(channel_dict_path, "rb") as f:
#         channel_dict = pickle.load(f)
    
    
#     model = Resnet_General(channel_dict)
#     model.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
#     return model




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
            
            pruned_weights[name] = weights[pruned_dim0][:,pruned_dim1,:,:]
    
    return pruned_info, num_pruned_channels, pruned_weights
            
            
def get_unpruned_indices_and_counts(model):
    non_pruned_info = {}
    num_unpruned_channels = {}
    unpruned_weights = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            
            layer_info = {}
            
            # Get the weight tensor
            weights = module.weight.detach()
            
            # Identify pruned filters along dim=0 (output channels)
            pruned_dim0 = torch.nonzero(weights.abs().sum(dim=(1, 2, 3)) > 0).squeeze().tolist()
            if isinstance(pruned_dim0, int):  # Handle single index case
                pruned_dim0 = [pruned_dim0]
            layer_info['pruned_dim0'] = pruned_dim0
            
            
            # Identify pruned filters along dim=1 (input channels)
            pruned_dim1 = torch.nonzero(weights.abs().sum(dim=(0, 2, 3)) > 0).squeeze().tolist()
            if isinstance(pruned_dim1, int):  # Handle single index case
                pruned_dim1 = [pruned_dim1]
            layer_info['pruned_dim1'] = pruned_dim1
        
            non_pruned_info[name] = layer_info
            num_unpruned_channels[name] = (len(pruned_dim0), len(pruned_dim1))
            unpruned_weights[name] = weights[pruned_dim0][:,pruned_dim1,:,:]
                            
                        
    return non_pruned_info, num_unpruned_channels, unpruned_weights

            

def extend_channels(model, pruned_dict):
    
    new_channel_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            
            new_in_channel = int(module.weight.data.shape[0] + pruned_dict[name][0])
            new_out_channel = int(module.weight.data.shape[1] + pruned_dict[name][1])
            
            new_channel_dict[name] = (new_in_channel, new_out_channel)
            
    return new_channel_dict



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
       
 
 
if __name__ == '__main__':

    model = SimpleCNN()
    Index_Aware_pruner(model, amount=0.2, plot=False)
    non_zero_weights, new_channels = get_nonzero_indices(model)
    print(new_channels)
    exit()
    
    # Example usage
    conv_layer = torch.nn.Conv2d(3, 6, kernel_size=3, bias=True)

    # Example pruning (randomly zero out some channels for demonstration)
    with torch.no_grad():
        conv_layer.weight[0] = 0  # Set all weights in channel 0 to zero
        conv_layer.weight[2] = 0  # Set all weights in channel 2 to zero


    # Prune conv1's output channels and use the same indices to prune conv2's input channels
    pruned_indices_conv1 = prune_conv_layer_dim0(model, 'conv1', amount=0.2, norm_type="L1")
    prune_conv_layer_dim1(model, 'conv2', pruned_indices_conv1)

    # Prune conv2's output channels and use the same indices to prune conv3's input channels
    pruned_indices_conv2 = prune_conv_layer_dim0(model, 'conv2', amount=0.2, norm_type="L1")
    prune_conv_layer_dim1(model, 'conv3', pruned_indices_conv2)
    
    pruned_indices_conv3 = prune_conv_layer_dim0(model, 'conv3', amount=0.2, norm_type="L1")








