import torch_pruning as tp
import torch
import copy
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn

from models.depGraph_fineTuner import DepGraphFineTuner
from utils.data_utils import load_data
from utils.eval_utils import evaluate_model
from utils.plot_utils import plot_metrics
from utils.device_utils import get_device
from utils.pruning_analysis import count_parameters, get_pruned_info, get_unpruned_info, extend_channels, AlexNet_General, calculate_last_conv_out_features, get_core_weights, reconstruct_weights_from_dicts, freeze_channels, debug_pruning_info, AlexNetLightningModule, model_size_in_mb

# def prune_model(original_model, model, device, pruning_percentage=0.2):
#     pruned_info = {}
    
#     model = model.to(device)
#     example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

#     # print("MODEL BEFORE PRUNING:\n", model.model)

#     DG = tp.DependencyGraph().build_dependency(model.model, example_inputs)
#     layers_to_prune = {
#         "model.features.3": model.model.features[3],
#         "model.features.6": model.model.features[6],
#         "model.features.8": model.model.features[8],
#         "model.features.10": model.model.features[10],
#     }


#     def get_pruning_indices(module, percentage):
#         with torch.no_grad():
#             weight = module.weight.data
#             if isinstance(module, torch.nn.Conv2d):
#                 channel_norms = weight.abs().mean(dim=[1,2,3])  
#             else:
#                 return None

#             pruning_count = int(channel_norms.size(0) * percentage)
#             if pruning_count == 0:
#                 return []
#             _, prune_indices = torch.topk(channel_norms, pruning_count, largest=False)
#             return prune_indices.tolist()
 
#     groups = []
#     for layer_name, layer_module in layers_to_prune.items():
#         if isinstance(layer_module, torch.nn.Conv2d):
#             prune_fn = tp.prune_conv_out_channels
#         else:
#             print(f"Skipping {layer_name}: Unsupported layer type {type(layer_module)}")
#             continue
        
#         pruning_idxs = get_pruning_indices(layer_module, pruning_percentage)
#         if pruning_idxs is None or len(pruning_idxs) == 0:
#             print(f"No channels to prune for {layer_name}.")
#             continue

#         group = DG.get_pruning_group(layer_module, prune_fn, idxs=pruning_idxs)
#         # print("group.details()", group.details()) 
#         if DG.check_pruning_group(group):
#             groups.append((layer_name, group))
#         else:
#             print(f"Invalid pruning group for layer {layer_name}, skipping pruning.")

#     if groups:
#         print(f"Pruning with {pruning_percentage*100}% percentage on {len(groups)} layers...")
#         for layer_name, group in groups:
#             print(f"Pruning layer: {layer_name}")
#             group.prune()

#         print("MODEL AFTER PRUNING:\n", model.model)
#     else:
#         print("No valid pruning groups found. The model was not pruned.")

#     # Check for all the pruned and unpruned indices and weights    
#     pruned_info, num_pruned_channels, pruned_weights = get_pruned_info(groups, original_model)
#     unpruned_info, num_unpruned_channels, unpruned_weights = get_unpruned_info(groups, original_model)

#     pruned_and_unpruned_info = {"pruned_info": pruned_info, 
#                                 "num_pruned_channels": num_pruned_channels, 
#                                 "pruned_weights": pruned_weights, 
#                                 "unpruned_info": unpruned_info, 
#                                 "num_unpruned_channels": num_unpruned_channels, 
#                                 "unpruned_weights": unpruned_weights}
#     return model, pruned_and_unpruned_info

def prune_model(original_model, model, device, layer_pruning_percentages):
    pruned_info = {}
    
    model = model.to(device)
    example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

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
                channel_norms = weight.abs().mean(dim=[1, 2, 3])  
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
        
        # Use layer-specific pruning percentage
        pruning_percentage = layer_pruning_percentages.get(layer_name, 0.2)  # Default to 20% if not specified
        pruning_idxs = get_pruning_indices(layer_module, pruning_percentage)
        if pruning_idxs is None or len(pruning_idxs) == 0:
            print(f"No channels to prune for {layer_name}.")
            continue

        group = DG.get_pruning_group(layer_module, prune_fn, idxs=pruning_idxs)
        if DG.check_pruning_group(group):
            groups.append((layer_name, group))
        else:
            print(f"Invalid pruning group for layer {layer_name}, skipping pruning.")

    if groups:
        print(f"Pruning with specified percentages on {len(groups)} layers...")
        for layer_name, group in groups:
            print(f"Pruning layer: {layer_name} with {layer_pruning_percentages[layer_name]*100}%")
            group.prune()

        print("MODEL AFTER PRUNING:\n", model.model)
    else:
        print("No valid pruning groups found. The model was not pruned.")

    pruned_info, num_pruned_channels, pruned_weights = get_pruned_info(groups, original_model)
    unpruned_info, num_unpruned_channels, unpruned_weights = get_unpruned_info(groups, original_model)

    pruned_and_unpruned_info = {
        "pruned_info": pruned_info, 
        "num_pruned_channels": num_pruned_channels, 
        "pruned_weights": pruned_weights, 
        "unpruned_info": unpruned_info, 
        "num_unpruned_channels": num_unpruned_channels, 
        "unpruned_weights": unpruned_weights
    }
    return model, pruned_and_unpruned_info


def main():
    wandb.init(project='alexnet_depGraph', name='AlexNet_Prune_Run')
    wandb_logger = WandbLogger(log_model=False)

    device = get_device()
    checkpoint_path = "./checkpoints/best_checkpoint_preTrained.ckpt"

    model = DepGraphFineTuner.load_from_checkpoint(checkpoint_path).to(device)

    layer_pruning_percentages = {
        "model.features.3": 0.15,  # Slight pruning for a sensitive layer
        "model.features.6": 0.3,   # Moderate pruning
        "model.features.8": 0.5,   # Aggressive pruning for less sensitive layers
        "model.features.10": 0.2,  # Moderate pruning for final layers
    }

    metrics_pruned = {
        "pruning_percentage": [],
        "test_accuracy": [],
        "f1_score": [],
        "count_params": [],
        "model_size": []
    }

    metrics_rebuild = {
        "pruning_percentage": [],
        "test_accuracy": [],
        "f1_score": [],
        "count_params": [],
        "model_size": []
    }

    train_dataloader, val_dataloader, test_dataloader = load_data(data_dir='./data', batch_size=32, val_split=0.2)
    # pruning_percentages = [0.2, 0.4, 0.6, 0.8]
    pruning_percentages = [0.2]

    trainer = pl.Trainer(max_epochs=5 , logger=wandb_logger, accelerator=device.type)

    print("MODEL BEFORE PRUNING:\n", model.model)
    orig_params = count_parameters(model)
    print(f"Original number of parameters: {orig_params}")

    # Evaluate before pruning
    orig_accuracy, orig_f1 = evaluate_model(model, test_dataloader, device)
    print(f"Original Accuracy: {orig_accuracy:.4f}, Original F1 Score: {orig_f1:.4f}")
    print("Model size in mb", model_size_in_mb(model))

    for pruning_percentage in pruning_percentages:
        print(f"Applying {pruning_percentage * 100}% pruning...")
        model_to_be_pruned = copy.deepcopy(model)
        # Prune the model
        core_model, pruned_and_unpruned_info = prune_model(model.model, model_to_be_pruned, device, layer_pruning_percentages=layer_pruning_percentages)
        # core_model, pruned_and_unpruned_info = prune_model(model.model, model_to_be_pruned, device, pruning_percentage=pruning_percentage)
        core_model = core_model.to(device)

        # Count parameters after pruning
        pruned_params = count_parameters(core_model)
        print(f"Number of parameters after pruning: {pruned_params}")
        print(f"Parameters reduced by: {orig_params - pruned_params} ({((orig_params - pruned_params) / orig_params) * 100:.2f}%)")

        pruned_accuracy, pruned_f1 = evaluate_model(core_model, test_dataloader, device)
        print(f"Accuracy immediately after pruning: {pruned_accuracy:.4f}, Pruned F1 Score: {pruned_f1:.4f}")

        pruned_model_size = model_size_in_mb(core_model)
        print("Model size in mb", pruned_model_size)
        # Fine-tune the pruned model using the method from DepGraphFineTuner
        if train_dataloader is not None and val_dataloader is not None:
            print("Starting post-pruning fine-tuning of the pruned model...")
            core_model.fine_tune_model(train_dataloader, val_dataloader, epochs=5, learning_rate=1e-4)

        pruned_accuracy, pruned_f1 = evaluate_model(core_model, test_dataloader, device)
        print(f"Accuracy after pruning and fine-tuning: {pruned_accuracy:.4f}, Pruned F1 Score: {pruned_f1:.4f}")

        # debug_pruning_info(model, core_model, pruned_and_unpruned_info["num_pruned_channels"], pruned_and_unpruned_info["num_unpruned_channels"])

        new_channels = extend_channels(core_model, pruned_and_unpruned_info["num_pruned_channels"])
        
        last_conv_out_features, last_conv_shape = calculate_last_conv_out_features(model.model)
        print(f"Last Conv Out Features: {last_conv_out_features}")
        print(f"Last Conv Shape: {last_conv_shape}")

        rebuilt_model = AlexNet_General(new_channels, last_conv_shape).to(device)
        get_core_weights(core_model, pruned_and_unpruned_info["unpruned_weights"])

        rebuilt_model = reconstruct_weights_from_dicts(rebuilt_model, pruned_indices=pruned_and_unpruned_info["pruned_info"], pruned_weights=pruned_and_unpruned_info["pruned_weights"], unpruned_indices=pruned_and_unpruned_info["unpruned_info"], unpruned_weights=pruned_and_unpruned_info["unpruned_weights"])
        rebuilt_model = freeze_channels(rebuilt_model, pruned_and_unpruned_info["unpruned_info"])

        rebuilt_model = rebuilt_model.to(device).to(torch.float32)
        print(rebuilt_model)

        rebuild_accuracy, rebuild_f1 = evaluate_model(rebuilt_model, test_dataloader, device)
        print(f"Accuracy after rebuilding: {rebuild_accuracy:.4f}, Pruned F1 Score: {rebuild_f1:.4f}")

        rebuild_model_size = model_size_in_mb(rebuilt_model)
        print("Model size in mb", rebuild_model_size)

        # Fine-tune the pruned model using the method from DepGraphFineTuner
        if train_dataloader is not None and val_dataloader is not None:
            print("Starting post-rebuilding fine-tuning of the pruned model...")
            rebuilt_model.fine_tune_model(train_dataloader, val_dataloader, device, epochs=5, learning_rate=1e-4)

        
        # Test the pruned model
        print("FINE TUNING COMPLETE")
        
        lightning_model = AlexNetLightningModule(rebuilt_model)
        trainer.test(lightning_model, dataloaders=test_dataloader)


        rebuild_accuracy, rebuild_f1 = evaluate_model(rebuilt_model, test_dataloader, device)
        print(f"Accuracy after rebuilding and fine tuning: {rebuild_accuracy:.4f}, Pruned F1 Score: {rebuild_f1:.4f}")

        metrics_pruned["pruning_percentage"].append(pruning_percentage * 100)
        metrics_pruned["test_accuracy"].append(pruned_accuracy)
        metrics_pruned["f1_score"].append(pruned_f1)
        metrics_pruned["count_params"].append(
            sum(p.numel() for p in core_model.parameters() if p.requires_grad)
        )
        metrics_pruned['model_size'].append(pruned_model_size)


        metrics_rebuild["pruning_percentage"].append(pruning_percentage * 100)
        metrics_rebuild["test_accuracy"].append(rebuild_accuracy)
        metrics_rebuild["f1_score"].append(rebuild_f1)
        metrics_rebuild["count_params"].append(
            sum(p.numel() for p in rebuilt_model.parameters() if p.requires_grad)
        )
        metrics_rebuild['model_size'].append(rebuild_model_size)


        print("All Metrics----------->", metrics_pruned)
        print("All Metrics----------->", metrics_rebuild)

        rebuilt_model.zero_grad()
        rebuilt_model.to("cpu")

    wandb.finish()

if __name__ == "__main__":
    main()
