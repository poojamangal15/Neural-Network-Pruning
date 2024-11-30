import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from torchvision.models import resnet18
import torch_pruning as tp
import torch

model = resnet18(pretrained=True).eval()


DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224))
group = DG.get_pruning_group(model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9])
if DG.check_pruning_group(group):
    group.prune()

print("Shape of conv1 weights after pruning:-----------------------", model.conv1.weight.shape)
print("Group details---------------------", group.details())

# Create a dummy input
input_tensor = torch.randn(1, 3, 224, 224)

# Forward pass through the pruned model
output = model(input_tensor)
print("Output shape:---------------------------", output.shape)


base_macs, base_params = tp.utils.count_ops_and_params(model, torch.randn(1, 3, 224, 224))
print(f"MACs:--------------------- {base_macs / 1e9} G, Params: {base_params / 1e6} M")

print("Total parameters after pruning:----------------------", sum(p.numel() for p in model.parameters()))
