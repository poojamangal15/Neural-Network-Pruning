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
from utils.pruning_analysis import (
    count_parameters, 
    get_pruned_info, 
    get_unpruned_info, 
    extend_channels, 
    AlexNet_General, 
    calculate_last_conv_out_features, 
    get_core_weights, 
    reconstruct_weights_from_dicts, 
    freeze_channels, 
    debug_pruning_info, 
    AlexNetLightningModule, 
    model_size_in_mb
)

# ------------------------------
# 1) Existing prune_model function
# ------------------------------
def prune_model(original_model, model, device, pruning_percentage=0.2):
    pruned_info = {}
    
    # if hasattr(model, "model"):
    #     inner_model = model.model
    # else:
    #     inner_model = model

    # inner_model = inner_model.to(device)
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
        if DG.check_pruning_group(group):
            groups.append((layer_name, group))
        else:
            print(f"Invalid pruning group for layer {layer_name}, skipping pruning.")

    if groups:
        print(f"Pruning with {pruning_percentage*100}% on {len(groups)} layers...")
        for layer_name, group in groups:
            print(f"Pruning layer: {layer_name}")
            group.prune()
        print("MODEL AFTER PRUNING:\n", model.model)
    else:
        print("No valid pruning groups found. The model was not pruned.")

    # Gather info about which channels were pruned/unpruned
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

# ---------------------------------------
# 2) NEW: iterative_depgraph_pruning function
#    Adapted from your supervisor's approach
# ---------------------------------------
def iterative_depgraph_pruning(
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    device,
    prune_ratios=[0.2, 0.4],
    fine_tune_epochs=5,
    fine_tune_lr=1e-4,
    rebuild=True
):
    """
    Perform iterative pruning using DepGraph-based approach.
    - model: DepGraphFineTuner (unpruned, original).
    - prune_ratios: List of prune percentages. e.g. [0.2, 0.4].
    - fine_tune_epochs: # epochs to fine tune after each pruning step.
    - rebuild: Whether to rebuild the model after each pruning step or only at the end.
    """

    # Evaluate original
    orig_params = count_parameters(model)
    orig_acc, orig_f1 = evaluate_model(model, test_dataloader, device)
    print(f"[Iterative] Original: Params={orig_params}, Acc={orig_acc:.4f}, F1={orig_f1:.4f}")
    print(f"Model size in MB: {model_size_in_mb(model):.2f}")

    # We'll store results after each pruning iteration
    iteration_results = []
    iteration_results.append({
        "prune_ratio": 0.0,
        "acc": orig_acc,
        "f1": orig_f1,
        "params": orig_params,
        "size_mb": model_size_in_mb(model)
    })

    # Make a local copy so we don't mutate the original
    current_model = copy.deepcopy(model)

    for i, ratio in enumerate(prune_ratios, start=1):
        print(f"\n[Iterative] Pruning iteration {i} with ratio={ratio}")
        print("Current model----->", current_model)
        # 1) Prune
        pruned_model, p_info = prune_model(
            original_model=model.model,  # pass the .model if needed
            model=current_model,
            device=device,
            pruning_percentage=ratio
        )
        # 2) Evaluate after prune
        p_params = count_parameters(pruned_model)
        p_acc, p_f1 = evaluate_model(pruned_model, test_dataloader, device)
        print(f"Immediately after prune ratio={ratio*100:.0f}%, Acc={p_acc:.4f}, F1={p_f1:.4f}, params={p_params}")

        # 3) Fine tune
        print(f"[Iterative] Fine-tuning pruned model for {fine_tune_epochs} epochs.")
        # pruned_model.fine_tune_model(train_dataloader, val_dataloader, device, epochs=fine_tune_epochs, learning_rate=fine_tune_lr)

        # Evaluate after fine tuning
        p_acc, p_f1 = evaluate_model(pruned_model, test_dataloader, device)
        p_size_mb = model_size_in_mb(pruned_model)
        print(f"[Iterative] Post-prune + fine-tune ratio={ratio*100:.0f}%, Acc={p_acc:.4f}, F1={p_f1:.4f}, size={p_size_mb:.2f} MB")

        iteration_results.append({
            "prune_ratio": ratio,
            "acc": p_acc,
            "f1": p_f1,
            "params": p_params,
            "size_mb": p_size_mb
        })

        # 4) Rebuild logic
        if rebuild:
            print("[Iterative] Rebuilding model to incorporate pruned channels.")
            # Prepare new channel dict based on pruned model
            new_channels = extend_channels(pruned_model, p_info["num_pruned_channels"])
            last_conv_out_features, last_conv_shape = calculate_last_conv_out_features(model.model)
            
            # Rebuild
            rebuilt_model = AlexNet_General(new_channels, last_conv_shape).to(device)
            get_core_weights(pruned_model, p_info["unpruned_weights"])
            rebuilt_model = reconstruct_weights_from_dicts(
                rebuilt_model,
                pruned_indices=p_info["pruned_info"], 
                pruned_weights=p_info["pruned_weights"],
                unpruned_indices=p_info["unpruned_info"], 
                unpruned_weights=p_info["unpruned_weights"]
            )
            rebuilt_model = freeze_channels(rebuilt_model, p_info["unpruned_info"])
            rebuilt_model = rebuilt_model.to(device).to(torch.float32)
            print("Rebuilt model----->", rebuilt_model)
            
            # Evaluate the rebuilt
            rb_acc, rb_f1 = evaluate_model(rebuilt_model, test_dataloader, device)
            rb_params = count_parameters(rebuilt_model)
            rb_size_mb = model_size_in_mb(rebuilt_model)
            print(f"[Iterative] Rebuilt model: Acc={rb_acc:.4f}, F1={rb_f1:.4f}, params={rb_params}, size={rb_size_mb:.2f} MB")

            # Fine-tune the plain model
            # But since we want to keep .model references, re-wrap:
            new_rebuilt = DepGraphFineTuner()
            new_rebuilt.model = rebuilt_model   # <--- The key step
            new_rebuilt = new_rebuilt.to(device)

            # Fine-tune the rebuilt model
            new_rebuilt.fine_tune_model(train_dataloader, val_dataloader, device, epochs=fine_tune_epochs, learning_rate=fine_tune_lr)
            rb_acc, rb_f1 = evaluate_model(new_rebuilt, test_dataloader, device)
            print(f"[Iterative] Rebuilt + FineTuned: Acc={rb_acc:.4f}, F1={rb_f1:.4f}")
            
            # If you want to continue iterative pruning with the newly rebuilt model, update current_model
            current_model = new_rebuilt
            print("Returned rebuilt model")
        else:
            # If not rebuilding, keep pruning on the pruned_model
            current_model = pruned_model

    return iteration_results

# -----------------------------------------
# 3) The main function that calls iterative_depgraph_pruning
# -----------------------------------------
def main():
    wandb.init(project='alexnet_depGraph', name='AlexNet_Iterative_Prune')
    wandb_logger = WandbLogger(log_model=False)

    device = get_device()
    checkpoint_path = "./checkpoints/best_checkpoint_preTrained.ckpt"

    # Load your original DepGraphFineTuner
    model = DepGraphFineTuner.load_from_checkpoint(checkpoint_path).to(device)

    # Load data
    train_dataloader, val_dataloader, test_dataloader = load_data(data_dir='./data', batch_size=32, val_split=0.2)

    # Optional: Create a PyTorch Lightning Trainer for test calls
    trainer = pl.Trainer(max_epochs=5, logger=wandb_logger, accelerator=device.type)

    # Let's define some pruning ratios
    iterative_ratios = [0.2, 0.2]  # for example, 2 steps of 20% each

    # Call the iterative pruning function
    results = iterative_depgraph_pruning(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        prune_ratios=iterative_ratios,
        fine_tune_epochs=5,
        fine_tune_lr=1e-4,
        rebuild=True
    )

    print("\n[FINAL] Iterative Pruning Results:")
    for step_res in results:
        print(step_res)

    wandb.finish()


if __name__ == "__main__":
    main()
