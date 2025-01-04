import torch
import torch_pruning as tp
import copy
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from models.depGraph_fineTuner import DepGraphFineTuner
from utils.data_utils import load_data
from utils.eval_utils import evaluate_model
from utils.plot_utils import plot_metrics
from utils.device_utils import get_device
from utils.pruning_analysis import count_parameters, get_pruned_info

def prune_model(original_model, model, device, pruning_percentage=0.2):
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
        "model.classifier.1": model.model.classifier[1],
        "model.classifier.4": model.model.classifier[4],
    }


    def get_pruning_indices(module, percentage):
        with torch.no_grad():
            weight = module.weight.data
            if isinstance(module, torch.nn.Conv2d):
                channel_norms = weight.abs().mean(dim=[1,2,3])  
            elif isinstance(module, torch.nn.Linear):
                channel_norms = weight.abs().mean(dim=1)
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
        elif isinstance(layer_module, torch.nn.Linear):
            prune_fn = tp.prune_linear_out_channels
        else:
            print(f"Skipping {layer_name}: Unsupported layer type {type(layer_module)}")
            continue
        
        # print("PRUNE FN-------", prune_fn)
        pruning_idxs = get_pruning_indices(layer_module, pruning_percentage)
        if pruning_idxs is None or len(pruning_idxs) == 0:
            print(f"No channels to prune for {layer_name}.")
            continue

        group = DG.get_pruning_group(layer_module, prune_fn, idxs=pruning_idxs)
        # print("group", group)
        # print("group.details()", group.details()) 
        if DG.check_pruning_group(group):
            groups.append((layer_name, group))
        else:
            print(f"Invalid pruning group for layer {layer_name}, skipping pruning.")

    # print("Pruning Groups:", groups)
    if groups:
        print(f"Pruning with {pruning_percentage*100}% percentage on {len(groups)} layers...")
        for layer_name, group in groups:
            print(f"Pruning layer: {layer_name}")
            group.prune()

        print("MODEL AFTER PRUNING:\n", model.model)
    else:
        print("No valid pruning groups found. The model was not pruned.")
    
    # Get pruned information
    pruned_info, num_pruned_channels, pruned_weights = get_pruned_info(groups, original_model)

    # Print the pruned info
    # print("PRUNED INFO", pruned_info)
    # print("Num pruned channels", num_pruned_channels)
    print("PRUNED WEIGHTS------>", pruned_weights)

    # for layer, info in pruned_info.items():
    #     print(f"Layer: {layer}")
    #     print(f"  Pruned dim0 (output channels): {info['pruned_dim0']}")
    #     print(f"  Pruned dim1 (input channels): {info['pruned_dim1']}")

    # print(f"Type of pruned_info: {type(pruned_info)}")

    # pruning_info, num_pruned_channels, pruned_weights = get_dependency_pruned_info(model.model,layers_to_prune, pruned_info)

    # print("PRUNED INFO---------_>", pruning_info)
    # print("Num pruned channels", num_pruned_channels)
    # print("Pruned weights", pruned_weights)
    return model

def main():
    wandb.init(project='alexnet_depGraph', name='AlexNet_Prune_Run')
    wandb_logger = WandbLogger(log_model=False)

    device = get_device()
    checkpoint_path = "./checkpoints/best_checkpoint.ckpt"
    # Load DepGraphFineTuner instead of AlexNetFineTuner so we can call fine_tune_model later
    model = DepGraphFineTuner.load_from_checkpoint(checkpoint_path).to(device)

    metrics = {
        "pruning_percentage": [],
        "test_accuracy": [],
        "f1_score": [],
        "model_size": []
    }

    train_dataloader, val_dataloader, test_dataloader = load_data(data_dir='./data', batch_size=32, val_split=0.2)
    pruning_percentages = [0.2]

    trainer = pl.Trainer(max_epochs=1 , logger=wandb_logger, accelerator=device.type)

    for pruning_percentage in pruning_percentages:
        print(f"Applying {pruning_percentage * 100}% pruning...")
        model_to_be_pruned = copy.deepcopy(model)

         # Count parameters before pruning
        orig_params = count_parameters(model_to_be_pruned)
        print(f"Original number of parameters: {orig_params}")

        # Evaluate before pruning
        orig_accuracy, orig_f1 = evaluate_model(model_to_be_pruned, test_dataloader, device)
        print(f"Original Accuracy: {orig_accuracy:.4f}, Original F1 Score: {orig_f1:.4f}")

        # Prune the model
        model_to_be_pruned = prune_model(model.model, model_to_be_pruned, device, pruning_percentage=pruning_percentage)
        model_to_be_pruned = model_to_be_pruned.to(device)

        # Count parameters after pruning
        pruned_params = count_parameters(model_to_be_pruned)
        print(f"Number of parameters after pruning: {pruned_params}")
        print(f"Parameters reduced by: {orig_params - pruned_params} ({((orig_params - pruned_params) / orig_params) * 100:.2f}%)")

    
        # Fine-tune the pruned model using the method from DepGraphFineTuner
        if train_dataloader is not None and val_dataloader is not None:
            print("Starting post-pruning fine-tuning of the pruned model...")
            model_to_be_pruned.fine_tune_model(train_dataloader, val_dataloader, epochs=1, learning_rate=1e-5)

        # # Save the pruned model's state dictionary
        # pruned_model_path = f"./pruned_models/alexnet_pruned_{int(pruning_percentage * 100)}.pth"
        # torch.save(model_to_be_pruned.state_dict(), pruned_model_path)

        # # Save the pruned_info dictionary
        # pruned_info_path = f"./pruned_models/pruned_info_{int(pruning_percentage * 100)}.pt"
        # torch.save(pruned_info, pruned_info_path)

        # print(f"Pruned model saved to: {pruned_model_path}")
        # print(f"Pruned info saved to: {pruned_info_path}")
        
        rebuilt_model = rebuild_model(model_to_be_pruned, pruned_info)

        # Test the pruned model
        trainer.test(model_to_be_pruned, dataloaders=test_dataloader)

        pruned_accuracy, pruned_f1 = evaluate_model(model_to_be_pruned, test_dataloader, device)
        print(f"Pruned Accuracy: {pruned_accuracy:.4f}, Pruned F1 Score: {pruned_f1:.4f}")

        metrics["pruning_percentage"].append(pruning_percentage * 100)
        metrics["test_accuracy"].append(pruned_accuracy)
        metrics["f1_score"].append(pruned_f1)
        metrics["model_size"].append(
            sum(p.numel() for p in model_to_be_pruned.parameters() if p.requires_grad)
        )

        print("All Metrics----------->", metrics)

        model_to_be_pruned.zero_grad()
        model_to_be_pruned.to("cpu")
        pruned_model_path = f"./pruned_models/alexnet_pruned_{int(pruning_percentage * 100)}.pth"
        torch.save(model_to_be_pruned.state_dict(), pruned_model_path)
        torch.save(pruned_info, f"pruned_info_{int(pruning_percentage * 100)}.pt")

        print(f"Pruned model saved to: {pruned_model_path}")

    plot_metrics(metrics)
    wandb.finish()

if __name__ == "__main__":
    main()
