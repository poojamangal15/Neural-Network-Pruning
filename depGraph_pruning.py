import torch
import torch_pruning as tp
import copy
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from models.depGraph_fineTuner import DepGraphFineTuner  # Note: We now use DepGraphFineTuner
from utils.data_utils import load_data
from utils.eval_utils import evaluate_model
from utils.plot_utils import plot_metrics
from utils.device_utils import get_device

def prune_model(model, device, pruning_percentage=0.2):
    
    model = model.to(device)
    example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

    print("MODEL BEFORE PRUNING:\n", model.model)

    DG = tp.DependencyGraph().build_dependency(model.model, example_inputs)

    layers_to_prune = {
        "conv2": model.model.features[3],
        "conv3": model.model.features[6],
        "conv4": model.model.features[8],
        "conv5": model.model.features[10],
        "fc1": model.model.classifier[1],
        "fc2": model.model.classifier[4]
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
        print(f"Pruning with {pruning_percentage*100}% percentage on {len(groups)} layers...")
        for layer_name, group in groups:
            print(f"Pruning layer: {layer_name}")
            group.prune()

        print("MODEL AFTER PRUNING:\n", model.model)
    else:
        print("No valid pruning groups found. The model was not pruned.")

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
    pruning_percentages = [0.5]

    trainer = pl.Trainer(max_epochs=5 , logger=wandb_logger, accelerator=device.type)

    for pruning_percentage in pruning_percentages:
        print(f"Applying {pruning_percentage * 100}% pruning...")
        model_to_be_pruned = copy.deepcopy(model)

        # Evaluate before pruning
        orig_accuracy, orig_f1 = evaluate_model(model_to_be_pruned, test_dataloader, device)
        print(f"Original Accuracy: {orig_accuracy:.4f}, Original F1 Score: {orig_f1:.4f}")

        # Prune the model
        model_to_be_pruned = prune_model(model_to_be_pruned, device, pruning_percentage=pruning_percentage)
        model_to_be_pruned = model_to_be_pruned.to(device)

        # Fine-tune the pruned model using the method from DepGraphFineTuner
        # (if loaders are available)
        if train_dataloader is not None and val_dataloader is not None:
            print("Starting post-pruning fine-tuning of the pruned model...")
            model_to_be_pruned.fine_tune_model(train_dataloader, val_dataloader, epochs=5, learning_rate=1e-5)

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
        print(f"Pruned model saved to: {pruned_model_path}")

    plot_metrics(metrics)
    wandb.finish()

if __name__ == "__main__":
    main()
