import torch_pruning as tp
import torch
import copy
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from utils.alexNet_fineTuner import AlexNetFineTuner
import torch.nn as nn

from utils.data_utils import load_data
from utils.eval_utils import evaluate_model, count_parameters, model_size_in_mb
# from utils.device_utils import get_device
from utils.pruning_analysis import get_device, prune_model, get_pruned_info, get_unpruned_info, extend_channels, AlexNet_General, calculate_last_conv_out_features, get_core_weights, reconstruct_weights_from_dicts, freeze_channels, fine_tuner, soft_pruning, copy_weights_from_dict, fine_tuner_zerograd, reconstruct_Global_weights_from_dicts, AlexNet_General_core, high_level_pruner


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
    schedulers = 'cosine',
    rebuild=True
):
    """
    Perform iterative pruning using DepGraph-based approach.
    - model: AlexNetFineTuner (unpruned, original).
    - prune_ratios: List of prune percentages. e.g. [0.2, 0.4].
    - fine_tune_epochs: # epochs to fine tune after each pruning step.
    - rebuild: Whether to rebuild the model after each pruning step or only at the end.
    """

    orig_params = count_parameters(model)
    orig_accuracy = evaluate_model(model, test_dataloader, device)
    print("Initial accuracy", orig_accuracy)
    orig_model_size = model_size_in_mb(model)


    pruned_models_info = []
    current_model = copy.deepcopy(model)
    prev_model = copy.deepcopy(model)

    # # We'll store results after each pruning iteration
    iteration_results = []
    prune_meta_data = []
    unprune_meta_data = []
    iteration_results.append({
        "step": "original",
        "acc": orig_accuracy,
        "params": orig_params,
        "size_mb": orig_model_size
    })

    models = []

    models.append(copy.deepcopy(current_model))

    for i, ratio in enumerate(prune_ratios, start=1):
        print(f"\n[Forward] Iteration {i}/{len(prune_ratios)}, Prune ratio={ratio}")
        
        print("STEP RATIO------", i, ratio)
        # 1) Prune
        pruned_model, pruned_and_unpruned_info = high_level_pruner(
            original_model=prev_model,
            model=current_model,
            device=device,
            pruning_percentage=ratio
        )

        pruned_model = pruned_model.to(device)

        core_model = AlexNet_General_core(pruned_and_unpruned_info['num_unpruned_channels']).to(device)
        copy_weights_from_dict(core_model, pruned_and_unpruned_info['unpruned_weights'])
        print("coremodel", core_model)
        
        models.append(copy.deepcopy(core_model))
       # 2) Evaluate after pruning
        pruned_params = count_parameters(core_model)
        pruned_accuracy = evaluate_model(core_model, test_dataloader, device)
        pruned_model_size = model_size_in_mb(core_model)

        prune_meta_data.append({"pruned_info" : pruned_and_unpruned_info['pruned_info'], "num_pruned_channels" : pruned_and_unpruned_info['num_pruned_channels'], "pruned_weights" : pruned_and_unpruned_info['pruned_weights']}) 
        unprune_meta_data.append({"unpruned_info" : pruned_and_unpruned_info['unpruned_info'], "num_unpruned_channels" : pruned_and_unpruned_info['num_unpruned_channels'], "unpruned_weights" : pruned_and_unpruned_info['unpruned_weights']})

        current_model = core_model
        prev_model = copy.deepcopy(core_model)
    
        # 3) Fine tune
    print(f"[Iterative] Fine-tuning pruned model for {fine_tune_epochs} epochs.")
    fine_tuner(core_model, train_dataloader, val_dataloader, device, pruning_percentage = ratio, fineTuningType = "pruning", epochs=fine_tune_epochs, scheduler_type=schedulers, LR=fine_tune_lr)

    # Evaluate after fine tuning
    pruned_accuracy = evaluate_model(core_model, test_dataloader, device)
    pruned_model_size = model_size_in_mb(core_model)
    pm_params = count_parameters(core_model)

    iteration_results.append({
            "step": f"forward_{i}",
            "acc": pruned_accuracy,
            "params": pruned_params,
            "size_mb": pruned_model_size
        })

    # 4) Rebuild logic
    if rebuild:
        for step_idx in range(len(prune_ratios)-1, -1, -1):
            print(f"{'-'*20} Rebuilding Descendant Model Level-{step_idx+1} {'-'*20}")
            model_state_prev = models[step_idx]
            model_state_pruned = models[step_idx+1]

            # print("MODEL_state prev", model_state_prev)
            # print("MODEL_state pruned", model_state_pruned)
            new_channels = extend_channels(model_state_pruned, prune_meta_data[step_idx]["num_pruned_channels"])
            
            # b) Construct a fresh "AlexNet"
            rebuilt_model = AlexNet_General_core(new_channels).to(device)
            # print("temp rebuilt", rebuilt_model)
            # c) Copy core weights
            get_core_weights(model_state_pruned, unprune_meta_data[step_idx]["unpruned_weights"])

            # d) Reconstruct from pruned + unpruned
            rebuilt_model, freeze_dim0, freeze_dim1 = reconstruct_Global_weights_from_dicts(
                rebuilt_model,
                pruned_indices=prune_meta_data[step_idx]["pruned_info"],
                pruned_weights=prune_meta_data[step_idx]["pruned_weights"],
                unpruned_indices=unprune_meta_data[step_idx]["unpruned_info"],
                unpruned_weights=unprune_meta_data[step_idx]["unpruned_weights"]
            )

            rebuilt_model = rebuilt_model.to(device).to(torch.float32)
            print("Rebuilt model----->", rebuilt_model)

            # g) Fine-tune the rebuilt
            print("[Backward Rebuild] Fine-tuning rebuilt model...")
            fine_tuner_zerograd(rebuilt_model, train_dataloader, val_dataloader, freeze_dim0, freeze_dim1, device, pruning_percentage = prune_ratios[step_idx], fineTuningType="rebuild", epochs=fine_tune_epochs, scheduler_type=schedulers, LR=fine_tune_lr)

            # h) Evaluate after fine-tune
            rebuild_accuracy = evaluate_model(rebuilt_model, test_dataloader, device)
            rebuild_params = count_parameters(rebuilt_model)
            rebuild_model_size = model_size_in_mb(rebuilt_model)

            iteration_results.append({
                "step": f"backward_{step_idx+1}",
                "acc": rebuild_accuracy,
                "params": rebuild_params,
                "size_mb": rebuild_model_size
            })

    return iteration_results

# -----------------------------------------
# 3) The main function that calls iterative_depgraph_pruning
# -----------------------------------------
def main(schedulers, lrs, epochs):
    wandb.init(project='AlexNet_Iterative', name='AlexNet_Iterative')
    wandb_logger = WandbLogger(log_model=False)

    device = get_device()
    # Load your original AlexNetFineTuner
    checkpoint_path = "./checkpoints/best_checkpoint_preTrained.ckpt"

    # Load your original AlexNetFineTuner
    model = AlexNetFineTuner.load_from_checkpoint(checkpoint_path).to(device)

    # Load data
    train_dataloader, val_dataloader, test_dataloader = load_data(data_dir='./data', batch_size=32, val_split=0.2)

    # Let's define some pruning ratios
    iterative_ratios = [0.2, 0.2, 0.2] 

    # Call the iterative pruning function
    results = iterative_depgraph_pruning(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        prune_ratios=iterative_ratios,
        fine_tune_epochs=epochs,
        fine_tune_lr= lrs,
        schedulers= schedulers,
        rebuild=True
    )

    print("\n[FINAL] Iterative Pruning Results:")
    for step_res in results:
        print(step_res)

    wandb.finish()


if __name__ == "__main__":
    schedulers = ['cosine']
    # schedulers = ['cosine', 'step', 'exponential', 'cyclic', 'Default']
    lrs = [1e-4]
    epochs = [100]
    model_name = "AlexNet_Iterative"

    for sch in schedulers:
        for lr in lrs:
            for epoch in epochs:
                 main(schedulers=sch, lrs=lr, epochs=epoch)