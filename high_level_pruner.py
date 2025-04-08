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
from utils.pruning_analysis import get_device, prune_model,  get_pruned_info, get_unpruned_info, extend_channels, AlexNet_General, calculate_last_conv_out_features, get_core_weights, reconstruct_weights_from_dicts, freeze_channels, fine_tuner, high_level_pruner, reconstruct_Global_weights_from_dicts, fine_tuner_zerograd, high_level_prunerTaylor, hessian_based_pruner

def verify_reconstructed_weights(rebuilt_model, pruned_info, pruned_weights, unpruned_info, unpruned_weights):
    """
    Verifies that weights in the rebuilt model match the expected pruned and unpruned weights.

    Parameters:
    - rebuilt_model: The model that has been rebuilt and had weights reassigned.
    - pruned_info: dict mapping layer names to {'pruned_dim0': [...], 'pruned_dim1': [...]}
    - pruned_weights: dict of stored pruned weights per layer
    - unpruned_info: dict mapping layer names to {'unpruned_dim0': [...], 'unpruned_dim1': [...]}
    - unpruned_weights: dict of stored unpruned weights per layer
    """
    for name, layer in rebuilt_model.named_modules():
        if not isinstance(layer, nn.Conv2d):
            continue

        layer_weights = layer.weight.detach().cpu()

        if name in pruned_info and name in pruned_weights:
            pd0 = pruned_info[name]['pruned_dim0']
            pd1 = pruned_info[name]['pruned_dim1']
            if pruned_weights[name].numel() > 0:
                if pd0 and pd1:
                    inserted_slice = layer_weights[pd0][:, pd1, :, :]
                    expected_slice = pruned_weights[name].cpu()
                elif pd0:
                    inserted_slice = layer_weights[pd0]
                    expected_slice = pruned_weights[name].cpu()
                elif pd1:
                    inserted_slice = layer_weights[:, pd1]
                    expected_slice = pruned_weights[name].cpu()
                else:
                    continue

                if torch.allclose(inserted_slice, expected_slice, atol=1e-6):
                    print(f"[Match ✅] Pruned weights correctly inserted for layer: {name}")
                else:
                    print(f"[Mismatch ❌] Pruned weights incorrect for layer: {name}")

        if name in unpruned_info and name in unpruned_weights:
            ud0 = unpruned_info[name]['unpruned_dim0']
            ud1 = unpruned_info[name]['unpruned_dim1']
            if unpruned_weights[name].numel() > 0:
                if ud0 and ud1:
                    inserted_slice = layer_weights[ud0][:, ud1, :, :]
                    expected_slice = unpruned_weights[name].cpu()
                elif ud0:
                    inserted_slice = layer_weights[ud0]
                    expected_slice = unpruned_weights[name].cpu()
                elif ud1:
                    inserted_slice = layer_weights[:, ud1]
                    expected_slice = unpruned_weights[name].cpu()
                else:
                    continue

                if torch.allclose(inserted_slice, expected_slice, atol=1e-6):
                    print(f"[Match ✅] Unpruned weights correctly inserted for layer: {name}")
                else:
                    print(f"[Mismatch ❌] Unpruned weights incorrect for layer: {name}")




def main(schedulers, lrs, epochs):
    print("ALEXNET HESSIAN PRUNING")
    wandb.init(project='alexnet_HighLevel', name='MagnitudeImportance')
    wandb_logger = WandbLogger(log_model=False)

    device = get_device()
    checkpoint_path = "./checkpoints/best_checkpoint_preTrained.ckpt"

    model = AlexNetFineTuner.load_from_checkpoint(checkpoint_path).to(device)
    print("MODEL BEFORE PRUNING:\n", model.model)

    pruning_percentages = [0.3, 0.5, 0.7]
    # pruning_percentages = [0.7]

    metrics_pruned = {
        "pruning_percentage": [], "LR": [], "scheduler": [], "epochs" : [], "test_accuracy": [], "count_params": [], "model_size": []
    }
    metrics_rebuild = {
        "pruning_percentage": [], "LR": [], "scheduler": [], "epochs" : [], "test_accuracy": [], "count_params": [], "model_size": []
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
        # core_model, pruned_and_unpruned_info = high_level_prunerTaylor(model.model, model_to_be_pruned, device, train_dataloader,pruning_percentage=pruning_percentage)
        # core_model, pruned_and_unpruned_info = high_level_pruner(model.model, model_to_be_pruned, device, pruning_percentage=pruning_percentage)
        core_model, pruned_and_unpruned_info = hessian_based_pruner(model, model_to_be_pruned, device, train_dataloader, pruning_percentage=pruning_percentage)

        core_model = core_model.to(device)
        print("core model", core_model)
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
        fine_tuner(core_model, train_dataloader, val_dataloader, device, pruning_percentage, fineTuningType = "pruning", epochs=epochs, scheduler_type=schedulers, LR=lrs)
        pruned_accuracy = evaluate_model(core_model, test_dataloader, device)

        wandb.log({
            "After Fine Tune Pruning Percentage": pruning_percentage * 100,
            "Test Accuracy (After Fine-Tuning)": pruned_accuracy,
        })

        new_channels = extend_channels(core_model, pruned_and_unpruned_info["num_pruned_channels"])        
        last_conv_out_features, last_conv_shape = calculate_last_conv_out_features(model.model)

        rebuilt_model = AlexNet_General(new_channels, last_conv_shape).to(device)
        get_core_weights(core_model, pruned_and_unpruned_info["unpruned_weights"])
        rebuilt_model, freeze_dim0, freeze_dim1 = reconstruct_Global_weights_from_dicts(rebuilt_model, pruned_indices=pruned_and_unpruned_info["pruned_info"], pruned_weights=pruned_and_unpruned_info["pruned_weights"], unpruned_indices=pruned_and_unpruned_info["unpruned_info"], unpruned_weights=pruned_and_unpruned_info["unpruned_weights"])
        verify_reconstructed_weights(
            rebuilt_model,
            pruned_info=pruned_and_unpruned_info["pruned_info"],
            pruned_weights=pruned_and_unpruned_info["pruned_weights"],
            unpruned_info=pruned_and_unpruned_info["unpruned_info"],
            unpruned_weights=pruned_and_unpruned_info["unpruned_weights"]
        )
        # rebuilt_model = freeze_channels(rebuilt_model, pruned_and_unpruned_info["unpruned_info"])
        rebuilt_model = rebuilt_model.to(device).to(torch.float32)

        rebuild_accuracy = evaluate_model(rebuilt_model, test_dataloader, device)
        rebuild_model_size = model_size_in_mb(rebuilt_model)

        wandb.log({
            "After Rebuild Pruning Percentage": pruning_percentage * 100,
            "Test Accuracy (After Rebuilding)": rebuild_accuracy,
            "Model Size (MB) After Rebuilding": rebuild_model_size
        })

        print("REBUILD MODEL", rebuilt_model)
        print("Starting post-rebuilding fine-tuning of the pruned model...")
        fine_tuner_zerograd(rebuilt_model, train_dataloader, val_dataloader, freeze_dim0, freeze_dim1, device, pruning_percentage, fineTuningType="rebuild", epochs=epochs, scheduler_type=schedulers, LR=5e-4)

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
        metrics_pruned['LR'].append(lrs)
        metrics_pruned['epochs'].append(epochs)

        metrics_rebuild["pruning_percentage"].append(pruning_percentage * 100)
        metrics_rebuild["scheduler"].append(schedulers)
        metrics_rebuild["test_accuracy"].append(rebuild_accuracy)
        metrics_rebuild["count_params"].append(
            sum(p.numel() for p in rebuilt_model.parameters() if p.requires_grad)
        )
        metrics_rebuild['model_size'].append(rebuild_model_size)
        metrics_rebuild['LR'].append(lrs)
        metrics_rebuild['epochs'].append(epochs)


        print("All Metrics for pruned model----------->", metrics_pruned)
        print("All Metrics for rebuild model----------->", metrics_rebuild)


        rebuilt_model.zero_grad()
        rebuilt_model.to("cpu")
    wandb.finish()

if __name__ == "__main__":
    schedulers = ['cosine']
    # schedulers = ['cosine', 'step', 'exponential', 'cyclic', 'Default']
    lrs = [1e-4]
    epochs = [100]
    model_name = "ResNet20"

    for sch in schedulers:
        for lr in lrs:
            for epoch in epochs:
                 main(schedulers=sch, lrs=lr, epochs=epoch)
