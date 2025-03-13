import torch_pruning as tp
import torch
import copy
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn

from utils.data_utils import load_data
from utils.eval_utils import evaluate_model, count_parameters, model_size_in_mb
# from utils.device_utils import get_device
from utils.pruning_analysis import get_device, prune_model,  get_pruned_info, get_unpruned_info, extend_channels, calculate_last_conv_out_features, get_core_weights, reconstruct_weights_from_dicts, freeze_channels, fine_tuner, high_level_pruner, VGG_General


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
    - model: AlexNetFineTuner (unpruned, original).
    - prune_ratios: List of prune percentages. e.g. [0.2, 0.4].
    - fine_tune_epochs: # epochs to fine tune after each pruning step.
    - rebuild: Whether to rebuild the model after each pruning step or only at the end.
    """

    # Evaluate original
    orig_params = count_parameters(model)
    orig_acc = evaluate_model(model, test_dataloader, device)
    orig_size_mb = model_size_in_mb(model)


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
        "params": orig_params,
        "size_mb": orig_size_mb
    })

    models = []

    models.append(copy.deepcopy(current_model))

    for i, ratio in enumerate(prune_ratios, start=1):
        print(f"\n[Forward] Iteration {i}/{len(prune_ratios)}, Prune ratio={ratio}")
        

        # 1) Prune
        core_model, pruned_and_unpruned_info = high_level_pruner(
            original_model=prev_model,
            model=current_model,
            device=device,
            pruning_percentage=ratio
        )
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
        # core_model.fine_tune_model(train_dataloader, val_dataloader, device, epochs=fine_tune_epochs, learning_rate=fine_tune_lr)
    fine_tuner(core_model, train_dataloader, val_dataloader, device, ratio, fineTuningType = "pruning", epochs=fine_tune_epochs, scheduler_type="cosine", LR=fine_tune_lr)

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
            print(f"{'-'*20} Rebuilding Descendant Model Level-{i+1} {'-'*20}")
            model_state_prev = models[step_idx]
            model_state_pruned = models[step_idx+1]

            # print("MODEL_state prev", model_state_prev)
            # print("MODEL_state pruned", model_state_pruned)
            new_channels = extend_channels(model_state_pruned, prune_meta_data[step_idx]["num_pruned_channels"])
            # print("NEW CHANNELS-------------->", new_channels)
            # new_channels = get_rebuild_channels(unprune_meta_data[step_idx]["num_unpruned_channels"], prune_meta_data[step_idx]["num_pruned_channels"])

            # b) Construct a fresh "AlexNet_General"
            rebuilt_model = VGG_General(new_channels).to(device)
            # print("temp rebuilt", rebuilt_model)
            # c) Copy core weights
            get_core_weights(model_state_pruned, unprune_meta_data[step_idx]["unpruned_weights"])

            # d) Reconstruct from pruned + unpruned
            rebuilt_model = reconstruct_weights_from_dicts(
                rebuilt_model,
                pruned_indices=prune_meta_data[step_idx]["pruned_info"],
                pruned_weights=prune_meta_data[step_idx]["pruned_weights"],
                unpruned_indices=unprune_meta_data[step_idx]["unpruned_info"],
                unpruned_weights=unprune_meta_data[step_idx]["unpruned_weights"]
            )

            # e) Freeze channels if needed
            # rebuilt_model = freeze_channels(rebuilt_model, unprune_meta_data[step_idx]["unpruned_info"])
            rebuilt_model = rebuilt_model.to(device).to(torch.float32)
            print("Rebuilt model----->", rebuilt_model)
            # f) Evaluate the newly rebuilt
            rb_acc = evaluate_model(rebuilt_model, test_dataloader, device)
            rb_params = count_parameters(rebuilt_model)
            rb_size = model_size_in_mb(rebuilt_model)

            # g) Fine-tune the rebuilt
            print("[Backward Rebuild] Fine-tuning rebuilt model...")
            # rebuilt_model.fine_tune_model(train_dataloader, val_dataloader, device, epochs=fine_tune_epochs, learning_rate=fine_tune_lr)
            fine_tuner(rebuilt_model, train_dataloader, val_dataloader, device, ratio, fineTuningType="rebuild", epochs=fine_tune_epochs, scheduler_type='cosine', LR=fine_tune_lr)

            # h) Evaluate after fine-tune
            r_ft_acc = evaluate_model(rebuilt_model, test_dataloader, device)
            r_ft_size = model_size_in_mb(rebuilt_model)

            iteration_results.append({
                "step": f"backward_{step_idx+1}",
                "acc": r_ft_acc,
                "params": rb_params,
                "size_mb": r_ft_size
            })

    return iteration_results

# -----------------------------------------
# 3) The main function that calls iterative_depgraph_pruning
# -----------------------------------------
def main():
    wandb.init(project='ResNet_Iterative', name='ResNet_Iterative')
    wandb_logger = WandbLogger(log_model=False)

    device = get_device()
    # Load your original AlexNetFineTuner
    model = torch.hub.load( "chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True).to(device)
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
        fine_tune_epochs=25,
        fine_tune_lr=1e-3,
        rebuild=True
    )

    print("\n[FINAL] Iterative Pruning Results:")
    for step_res in results:
        print(step_res)

    wandb.finish()


if __name__ == "__main__":
    main()