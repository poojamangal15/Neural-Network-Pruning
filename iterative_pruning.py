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
    print("PRUNING PERCENTAGE------------------------------------>", pruning_percentage)

    print("CURRENT MODEL-------------------", model)
    model = model.to(device)
    example_inputs = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

    # print("MODEL BEFORE PRUNING:\n", model.model)

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

        print("MODEL AFTER PRUNING:\n", model.model)
    else:
        print("No valid pruning groups found. The model was not pruned.")

    print("ORIGINAL MODEL-------------------", original_model)
    print("PRUNED MODEL-------------------", model)
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


# Iterative pruning approach: Prune all steps then do a reverse rebuilding
def iterative_depgraph_pruning(
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    device,
    prune_ratios=[0.2, 0.2, 0.2],
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
    orig_size_mb = model_size_in_mb(model)

    print(f"[IterativeSup] Original => Acc={orig_acc:.4f}, F1={orig_f1:.4f}, params={orig_params}, size={orig_size_mb:.2f}MB")


    pruned_models_info = []
    current_model = copy.deepcopy(model)
    prev_model = copy.deepcopy(model)

    # # We'll store results after each pruning iteration
    iteration_results = []
    prune_meta_data = []
    unprune_meta_data = []
    iteration_results.append({
        "step": "original",
        "acc": orig_acc,
        "f1": orig_f1,
        "params": orig_params,
        "size_mb": orig_size_mb
    })

    for i, ratio in enumerate(prune_ratios, start=1):
        print(f"\n[Forward] Iteration {i}/{len(prune_ratios)}, Prune ratio={ratio}")
        print("Current model----->", current_model)

        # 1) Prune
        core_model, pruned_and_unpruned_info = prune_model(
            original_model=prev_model.model,
            model=current_model,
            device=device,
            pruning_percentage=ratio
        )

       # 2) Evaluate after pruning
        p_params = count_parameters(core_model)
        p_acc, p_f1 = evaluate_model(core_model, test_dataloader, device)
        p_size_mb = model_size_in_mb(core_model)
        print(f"[Forward] Post-prune => Acc={p_acc:.4f}, F1={p_f1:.4f}, params={p_params}, size={p_size_mb:.2f}MB")

        prune_meta_data.append({"zero_channels_info" : pruned_and_unpruned_info['pruned_info'], "num_pruned_channels" : pruned_and_unpruned_info['num_pruned_channels'], "pruned_weights" : pruned_and_unpruned_info['pruned_weights']}) 
        unprune_meta_data.append({"non_zero_channels_info" : pruned_and_unpruned_info['unpruned_info'], "num_unpruned_channels" : pruned_and_unpruned_info['num_unpruned_channels'], "unpruned_weights" : pruned_and_unpruned_info['unpruned_weights']})

        print("Pruning Metadata saved successfully.")
        current_model = core_model
        prev_model = copy.deepcopy(core_model)
    
    # 3) Fine tune
    print(f"[Iterative] Fine-tuning pruned model for {fine_tune_epochs} epochs.")
    core_model.fine_tune_model(train_dataloader, val_dataloader, device, epochs=fine_tune_epochs, learning_rate=fine_tune_lr)

    # Evaluate after fine tuning
    p_acc, p_f1 = evaluate_model(core_model, test_dataloader, device)
    p_size_mb = model_size_in_mb(core_model)
    print(f"[Iterative] Post-prune + fine-tune ratio={ratio*100:.0f}%, Acc={p_acc:.4f}, F1={p_f1:.4f}, size={p_size_mb:.2f} MB")

    iteration_results.append({
            "step": f"forward_{i}",
            "acc": ft_acc,
            "f1": ft_f1,
            "params": pm_params,
            "size_mb": ft_size
        })

    # 4) Rebuild logic
    if rebuild:
        for step_idx in range(len(prune_ratios)-1, -1, -1):
            ratio = prune_ratios[step_idx]
            print(f"\n=== BACKWARD REBUILD STEP {step_idx+1}, ratio={ratio} ===")

            # a) Merge channel info: (like your supervisor code, new_channels = unpruned + pruned)
            new_channels = extend_channels(rebuilt_model, prune_meta_data[step_idx]["num_pruned_channels"])
            # Or if your code does it differently: ...
            last_conv_out, last_conv_shape = calculate_last_conv_out_features(rebuilt_model.model)
            
            # b) Construct a fresh "AlexNet_General"
            temp_rebuilt = AlexNet_General(new_channels, last_conv_shape).to(device)
            
            # c) Copy core weights
            get_core_weights(rebuilt_model, unprune_meta_data[step_idx]["unpruned_weights"])

            # d) Reconstruct from pruned + unpruned
            temp_rebuilt = reconstruct_weights_from_dicts(
                temp_rebuilt,
                pruned_indices=prune_meta_data[step_idx]["pruned_info"],
                pruned_weights=prune_meta_data[step_idx]["pruned_weights"],
                unpruned_indices=unprune_meta_data[step_idx]["unpruned_info"],
                unpruned_weights=unprune_meta_data[step_idx]["unpruned_weights"]
            )

            # e) Freeze channels if needed
            temp_rebuilt = freeze_channels(temp_rebuilt, unprune_meta_data[step_idx]["unpruned_info"])

            # f) Evaluate the newly rebuilt
            rb_acc, rb_f1 = evaluate_model(temp_rebuilt, test_loader, device)
            rb_params = count_parameters(temp_rebuilt)
            rb_size = model_size_in_mb(temp_rebuilt)
            print(f"[Backward Rebuild] Step {step_idx+1} => Acc={rb_acc:.4f}, F1={rb_f1:.4f}, Params={rb_params}, Size={rb_size:.2f} MB")

            # g) Fine-tune the rebuilt
            print("[Backward Rebuild] Fine-tuning rebuilt model...")
            # temp_rebuilt.fine_tune_model(train_loader, val_loader, device, epochs=fine_tune_epochs, learning_rate=fine_tune_lr)

            # h) Evaluate after fine-tune
            r_ft_acc, r_ft_f1 = evaluate_model(temp_rebuilt, test_loader, device)
            r_ft_size = model_size_in_mb(temp_rebuilt)
            print(f"[Backward Rebuild] Post-fine-tune => Acc={r_ft_acc:.4f}, F1={r_ft_f1:.4f}, Size={r_ft_size:.2f} MB")

            iteration_results.append({
                "step": f"backward_{step_idx+1}",
                "acc": r_ft_acc,
                "f1": r_ft_f1,
                "params": rb_params,
                "size_mb": r_ft_size
            })

            # i) Update reference so we can continue "backward rebuild" from the newly built model
            rebuilt_model = temp_rebuilt

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
    iterative_ratios = [0.1, 0.2, 0.3]  # for example, 2 steps of 20% each

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
        rebuild=False
    )

    print("\n[FINAL] Iterative Pruning Results:")
    for step_res in results:
        print(step_res)

    wandb.finish()


if __name__ == "__main__":
    main()
