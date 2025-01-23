import torch
import torch_pruning as tp
import copy
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn

import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter
from torch.utils.data import random_split, DataLoader
from utils.data_utils import load_data
from utils.eval_utils import evaluate_model
from utils.plot_utils import plot_metrics
from utils.device_utils import get_device
from utils.pruning_analysis import count_parameters, get_pruned_info, get_unpruned_info, extend_channels, fine_tuner, calculate_last_conv_out_features, get_core_weights, reconstruct_weights_from_dicts, freeze_channels, debug_pruning_info, Resnet_General


train_transform = Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Data transformations for validation/test
test_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Split the train dataset into train/val
train_size = int((1 - 0.2) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=4)


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
    pruned_info, num_pruned_channels, pruned_weights = get_pruned_info(groups, original_model)
    unpruned_info, num_unpruned_channels, unpruned_weights = get_unpruned_info(groups, original_model)

    pruned_and_unpruned_info = {"pruned_info": pruned_info, 
                                "num_pruned_channels": num_pruned_channels, 
                                "pruned_weights": pruned_weights, 
                                "unpruned_info": unpruned_info, 
                                "num_unpruned_channels": num_unpruned_channels, 
                                "unpruned_weights": unpruned_weights}
    return model, pruned_and_unpruned_info

def main():
    wandb.init(project='resnet20_depGraph', name='ResNet20_Prune_Run')
    wandb_logger = WandbLogger(log_model=False)

    device = get_device()

    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_resnet20",
        pretrained=True
    ).to(device)

    metrics = {
        "pruning_percentage": [],
        "test_accuracy": [],
        "f1_score": [],
        "model_size": []
    }

    pruning_percentages = [0.2]

    trainer = pl.Trainer(max_epochs=5 , logger=wandb_logger, accelerator=device.type)

    for pruning_percentage in pruning_percentages:
        print(f"Applying {pruning_percentage * 100}% pruning...")
        model_to_be_pruned = copy.deepcopy(model)

         # Count parameters before pruning
        print("MODEL BEFORE PRUNING:\n", model)
        orig_params = count_parameters(model_to_be_pruned)
        print(f"Original number of parameters: {orig_params}")

        # Evaluate before pruning
        orig_accuracy, orig_f1 = evaluate_model(model_to_be_pruned, test_dataloader, device)
        print(f"Original Accuracy: {orig_accuracy:.4f}, Original F1 Score: {orig_f1:.4f}")

        # Prune the model
        core_model, pruned_and_unpruned_info = prune_model(model, model_to_be_pruned, device, pruning_percentage=pruning_percentage)
        core_model = core_model.to(device)

        # Count parameters after pruning
        pruned_params = count_parameters(core_model)
        print(f"Number of parameters after pruning: {pruned_params}")
        print(f"Parameters reduced by: {orig_params - pruned_params} ({((orig_params - pruned_params) / orig_params) * 100:.2f}%)")

    
        # Fine-tune the pruned model using the method from DepGraphFineTuner
        if train_dataloader is not None and val_dataloader is not None:
            print("Starting post-pruning fine-tuning of the pruned model...")
            scheduler_type = 'step'
            # fine_tuner(core_model, train_dataloader, val_dataloader, scheduler_type, device, num_epochs=5, LR=1e-3)

        pruned_accuracy, pruned_f1 = evaluate_model(core_model, test_dataloader, device)
        print(f"Pruned Accuracy: {pruned_accuracy:.4f}, Pruned F1 Score: {pruned_f1:.4f}")

        debug_pruning_info(model, core_model, pruned_and_unpruned_info["num_pruned_channels"], pruned_and_unpruned_info["num_unpruned_channels"])


        new_channels = extend_channels(core_model, pruned_and_unpruned_info["num_pruned_channels"])
        
        # last_conv_out_features, last_conv_shape = calculate_last_conv_out_features(model)
        # print(f"Last Conv Out Features: {last_conv_out_features}")
        # print(f"Last Conv Shape: {last_conv_shape}")

        rebuilt_model = Resnet_General(new_channels).to(device)
        get_core_weights(core_model, pruned_and_unpruned_info["unpruned_weights"])

        rebuilt_model = reconstruct_weights_from_dicts(rebuilt_model, pruned_indices=pruned_and_unpruned_info["pruned_info"], pruned_weights=pruned_and_unpruned_info["pruned_weights"], unpruned_indices=pruned_and_unpruned_info["unpruned_info"], unpruned_weights=pruned_and_unpruned_info["unpruned_weights"])
        rebuilt_model = freeze_channels(rebuilt_model, pruned_and_unpruned_info["unpruned_info"])

        rebuilt_model = rebuilt_model.to(device).to(torch.float32)
        print(rebuilt_model)

        # Fine-tune the pruned model using the method from DepGraphFineTuner
        if train_dataloader is not None and val_dataloader is not None:
            print("Starting post-rebuilding fine-tuning of the pruned model...")
            rebuilt_model.fine_tune_model(train_dataloader, val_dataloader, device, epochs=5, learning_rate=1e-4)
        
        # Test the pruned model
        print("FINE TUNING COMPLETE")
        
        lightning_model = AlexNetLightningModule(rebuilt_model)
        trainer.test(lightning_model, dataloaders=test_dataloader)


        pruned_accuracy, pruned_f1 = evaluate_model(rebuilt_model, test_dataloader, device)
        print(f"Rebuild Accuracy: {pruned_accuracy:.4f}, Pruned F1 Score: {pruned_f1:.4f}")

        metrics["pruning_percentage"].append(pruning_percentage * 100)
        metrics["test_accuracy"].append(pruned_accuracy)
        metrics["f1_score"].append(pruned_f1)
        metrics["model_size"].append(
            sum(p.numel() for p in rebuilt_model.parameters() if p.requires_grad)
        )

        print("All Metrics----------->", metrics)

        rebuilt_model.zero_grad()
        rebuilt_model.to("cpu")
        pruned_model_path = f"./pruned_models/alexnet_pruned_{int(pruning_percentage * 100)}.pth"
        torch.save(rebuilt_model.state_dict(), pruned_model_path)
        torch.save(pruned_and_unpruned_info, f"pruned_info_{int(pruning_percentage * 100)}.pt")

        print(f"Pruned model saved to: {pruned_model_path}")

    plot_metrics(metrics)
    wandb.finish()

if __name__ == "__main__":
    main()
