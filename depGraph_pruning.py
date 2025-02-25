import torch_pruning as tp
import torch
import copy
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn

from utils.alexNet_fineTuner import AlexNetFineTuner
from utils.data_utils import load_data
from utils.eval_utils import evaluate_model, count_parameters, model_size_in_mb
# from utils.device_utils import get_device
from utils.pruning_analysis import get_device, prune_model,  get_pruned_info, get_unpruned_info, extend_channels, AlexNet_General, calculate_last_conv_out_features, get_core_weights, reconstruct_weights_from_dicts, freeze_channels, fine_tuner



def main(schedulers):
    wandb.init(project='alexnet_depGraph', name='AlexNet_Prune_Run')
    wandb_logger = WandbLogger(log_model=False)

    device = get_device()
    checkpoint_path = "./checkpoints/best_checkpoint_preTrained.ckpt"

    model = AlexNetFineTuner.load_from_checkpoint(checkpoint_path).to(device)
    print("MODEL BEFORE PRUNING:\n", model.model)

    # pruning_percentages = [0.2, 0.4, 0.6, 0.8]
    pruning_percentages = [0.2]

    metrics_pruned = {
        "pruning_percentage": [], "scheduler": [], "test_accuracy": [], "count_params": [], "model_size": []
    }
    metrics_rebuild = {
        "pruning_percentage": [], "scheduler": [], "test_accuracy": [], "count_params": [], "model_size": []
    }

    train_dataloader, val_dataloader, test_dataloader = load_data(data_dir='./data', batch_size=32, val_split=0.2)

    orig_params = count_parameters(model)
    orig_accuracy = evaluate_model(model, test_dataloader, device)
    print("Initial accuracy", orig_accuracy)
    pruned_model_size = model_size_in_mb(model)


    for pruning_percentage in pruning_percentages:
        print(f"Applying {pruning_percentage * 100}% pruning...")
        model_to_be_pruned = copy.deepcopy(model)
        # Prune the model
        core_model, pruned_and_unpruned_info = prune_model(model.model, model_to_be_pruned, device, pruning_percentage=pruning_percentage)
        core_model = core_model.to(device)

        # Count parameters after pruning
        pruned_params = count_parameters(core_model)
        pruned_accuracy = evaluate_model(core_model, test_dataloader, device)
        pruned_model_size = model_size_in_mb(core_model)

        wandb.log({
            "Pruning Percentage": pruning_percentage * 100,
            "Test Accuracy (After Pruning)": pruned_accuracy,
            "Model Size (MB) After Pruning": pruned_model_size,
            "Params Reduced (%)": orig_params - pruned_params
        })

        print("Starting post-pruning fine-tuning of the pruned model...")
        fine_tuner(core_model, train_dataloader, val_dataloader, device, fineTuningType = "pruning", epochs=5, scheduler_type=schedulers, LR=1e-4)
        pruned_accuracy = evaluate_model(core_model, test_dataloader, device)

        wandb.log({
            "Pruning Percentage": pruning_percentage * 100,
            "Test Accuracy (After Fine-Tuning)": pruned_accuracy,
        })

        new_channels = extend_channels(core_model, pruned_and_unpruned_info["num_pruned_channels"])        
        last_conv_out_features, last_conv_shape = calculate_last_conv_out_features(model.model)

        rebuilt_model = AlexNet_General(new_channels, last_conv_shape).to(device)
        get_core_weights(core_model, pruned_and_unpruned_info["unpruned_weights"])
        rebuilt_model = reconstruct_weights_from_dicts(rebuilt_model, pruned_indices=pruned_and_unpruned_info["pruned_info"], pruned_weights=pruned_and_unpruned_info["pruned_weights"], unpruned_indices=pruned_and_unpruned_info["unpruned_info"], unpruned_weights=pruned_and_unpruned_info["unpruned_weights"])
        # rebuilt_model = freeze_channels(rebuilt_model, pruned_and_unpruned_info["unpruned_info"])
        rebuilt_model = rebuilt_model.to(device).to(torch.float32)

        rebuild_accuracy = evaluate_model(rebuilt_model, test_dataloader, device)
        rebuild_model_size = model_size_in_mb(rebuilt_model)

        wandb.log({
            "Pruning Percentage": pruning_percentage * 100,
            "Test Accuracy (After Rebuilding)": rebuild_accuracy,
            "Model Size (MB) After Rebuilding": rebuild_model_size
        })

        print("Starting post-rebuilding fine-tuning of the pruned model...")
        fine_tuner(rebuilt_model, train_dataloader, val_dataloader, device, fineTuningType="rebuild", epochs=5, scheduler_type=schedulers, LR=1e-4)

        rebuild_accuracy = evaluate_model(rebuilt_model, test_dataloader, device)

        wandb.log({
            "Pruning Percentage": pruning_percentage * 100,
            "Test Accuracy (After Rebuilding & Fine-Tuning)": rebuild_accuracy,
        })

        metrics_pruned["pruning_percentage"].append(pruning_percentage * 100)
        metrics_pruned["scheduler"].append(schedulers)
        metrics_pruned["test_accuracy"].append(pruned_accuracy)
        metrics_pruned["count_params"].append(
            sum(p.numel() for p in core_model.parameters() if p.requires_grad)
        )
        metrics_pruned['model_size'].append(pruned_model_size)


        metrics_rebuild["pruning_percentage"].append(pruning_percentage * 100)
        metrics_rebuild["scheduler"].append(schedulers)
        metrics_rebuild["test_accuracy"].append(rebuild_accuracy)
        metrics_rebuild["count_params"].append(
            sum(p.numel() for p in rebuilt_model.parameters() if p.requires_grad)
        )
        metrics_rebuild['model_size'].append(rebuild_model_size)


        print("All Metrics for pruned model----------->", metrics_pruned)
        print("All Metrics for rebuild model----------->", metrics_rebuild)

        rebuilt_model.zero_grad()
        rebuilt_model.to("cpu")
    wandb.finish()

if __name__ == "__main__":
    schedulers = ['cosine']
    # schedulers = ['cosine', 'step', 'exponential', 'cyclic']
    for sch in schedulers:
        main(schedulers=sch)
