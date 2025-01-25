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

def iterative_pruning(
    model,
    original_model,
    device,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    pruning_steps=3,
    pruning_ratios=[0.3, 0.3, 0.3],
    epochs=5,
    learning_rate=1e-4
):
    """
    Perform iterative pruning for a specified number of steps.
    Each step does:
      1. Prune the model by `pruning_ratios[i]`.
      2. Fine-tune the pruned model (the 'core' model).
      3. (Optional) Rebuild the model from pruned + unpruned info, fine-tune it again.

    :param model: DepGraphFineTuner loaded model (unpruned).
    :param original_model: A copy of the original model used for reference in get_pruned_info.
    :param device: Torch device (cuda/cpu).
    :param train_dataloader: Dataloader for training.
    :param val_dataloader: Dataloader for validation.
    :param test_dataloader: Dataloader for testing.
    :param pruning_steps: Number of iterations of pruning.
    :param pruning_ratios: List of floats for each pruning step (e.g. [0.3, 0.3, 0.3]).
    :param epochs: Number of epochs to fine-tune at each step.
    :param learning_rate: Learning rate for fine-tuning.
    """

    # Make sure the ratio list matches the number of steps
    assert pruning_steps == len(pruning_ratios), (
        f"pruning_steps ({pruning_steps}) must match length of pruning_ratios ({len(pruning_ratios)})"
    )

    # We will store metrics after each step for pruned & rebuilt models
    metrics_pruned = {
        "pruning_step": [],
        "pruning_ratio": [],
        "test_accuracy": [],
        "f1_score": [],
        "count_params": [],
        "model_size_mb": []
    }

    metrics_rebuilt = {
        "pruning_step": [],
        "pruning_ratio": [],
        "test_accuracy": [],
        "f1_score": [],
        "count_params": [],
        "model_size_mb": []
    }

    # Make a fresh copy of the model to prune iteratively
    iterative_model = copy.deepcopy(model)
    iterative_model = iterative_model.to(device)

    orig_params = count_parameters(iterative_model)
    print(f"Original param count: {orig_params}")

    # Evaluate before ANY pruning:
    orig_acc, orig_f1 = evaluate_model(iterative_model, test_dataloader, device)
    print(f"Original Acc: {orig_acc:.4f}, F1: {orig_f1:.4f}")

    # Start iterative pruning
    for step_idx in range(pruning_steps):
        print(f"\n=== Pruning Step {step_idx+1}/{pruning_steps}, Ratio = {pruning_ratios[step_idx]*100:.1f}% ===")

        # 1) Prune the model
        pruned_model, pruned_info = prune_model(
            original_model=original_model.model,   # The original reference for get_pruned_info
            model=iterative_model,                 # The model to actually prune
            device=device,
            pruning_percentage=pruning_ratios[step_idx]
        )

        # 2) Evaluate the model immediately after pruning
        pruned_params = count_parameters(pruned_model)
        pruned_acc, pruned_f1 = evaluate_model(pruned_model, test_dataloader, device)
        size_pruned_mb = model_size_in_mb(pruned_model)

        print(f"After Pruning Step {step_idx+1} => Params: {pruned_params}, Acc: {pruned_acc:.4f}, F1: {pruned_f1:.4f}, size: {size_pruned_mb:.2f} MB")

        # 3) Fine-tune the pruned model
        if train_dataloader and val_dataloader:
            print("Fine-tuning pruned model...")
            pruned_model.fine_tune_model(train_dataloader, val_dataloader, epochs=epochs, learning_rate=learning_rate)

        # Evaluate after fine-tuning
        pruned_acc, pruned_f1 = evaluate_model(pruned_model, test_dataloader, device)
        size_pruned_mb = model_size_in_mb(pruned_model)
        print(f"Fine-tuned Pruned => Acc: {pruned_acc:.4f}, F1: {pruned_f1:.4f}, size: {size_pruned_mb:.2f} MB")

        # Save pruned model stats to dictionary
        metrics_pruned["pruning_step"].append(step_idx+1)
        metrics_pruned["pruning_ratio"].append(pruning_ratios[step_idx])
        metrics_pruned["test_accuracy"].append(pruned_acc)
        metrics_pruned["f1_score"].append(pruned_f1)
        metrics_pruned["count_params"].append(pruned_params)
        metrics_pruned["model_size_mb"].append(size_pruned_mb)

        # 4) Rebuild the model from pruned + unpruned info (like your code does)
        #    The pruned_and_unpruned_info returned by prune_model is in `pruned_info`.
        new_channels = extend_channels(pruned_model, pruned_info["num_pruned_channels"])
        last_conv_out_features, last_conv_shape = calculate_last_conv_out_features(iterative_model.model)  # or pruned_model.model
        rebuilt_model = AlexNet_General(new_channels, last_conv_shape).to(device)
        # fill in unpruned weights
        get_core_weights(pruned_model, pruned_info["unpruned_weights"])
        # reconstruct pruned + unpruned
        rebuilt_model = reconstruct_weights_from_dicts(
            rebuilt_model,
            pruned_indices=pruned_info["pruned_info"],
            pruned_weights=pruned_info["pruned_weights"],
            unpruned_indices=pruned_info["unpruned_info"],
            unpruned_weights=pruned_info["unpruned_weights"]
        )
        # freeze channels
        rebuilt_model = freeze_channels(rebuilt_model, pruned_info["unpruned_info"])
        rebuilt_model = rebuilt_model.to(device).to(torch.float32)

        # Evaluate immediately after rebuild
        rebuilt_acc, rebuilt_f1 = evaluate_model(rebuilt_model, test_dataloader, device)
        rebuilt_params = count_parameters(rebuilt_model)
        size_rebuilt_mb = model_size_in_mb(rebuilt_model)
        print(f"Rebuilt => Acc: {rebuilt_acc:.4f}, F1: {rebuilt_f1:.4f}, size: {size_rebuilt_mb:.2f} MB")

        # 5) Fine-tune the rebuilt model
        if train_dataloader and val_dataloader:
            print("Fine-tuning rebuilt model...")
            rebuilt_model.fine_tune_model(train_dataloader, val_dataloader, device, epochs=epochs, learning_rate=learning_rate)

        rebuilt_acc, rebuilt_f1 = evaluate_model(rebuilt_model, test_dataloader, device)
        rebuilt_params = count_parameters(rebuilt_model)
        size_rebuilt_mb = model_size_in_mb(rebuilt_model)
        print(f"Fine-tuned Rebuilt => Acc: {rebuilt_acc:.4f}, F1: {rebuilt_f1:.4f}, size: {size_rebuilt_mb:.2f} MB\n")

        # Save rebuilt model stats to dictionary
        metrics_rebuilt["pruning_step"].append(step_idx+1)
        metrics_rebuilt["pruning_ratio"].append(pruning_ratios[step_idx])
        metrics_rebuilt["test_accuracy"].append(rebuilt_acc)
        metrics_rebuilt["f1_score"].append(rebuilt_f1)
        metrics_rebuilt["count_params"].append(rebuilt_params)
        metrics_rebuilt["model_size_mb"].append(size_rebuilt_mb)

        # Update the "iterative_model" to continue pruning next step from the "rebuilt_model"
        # or from the "core" model. 
        # Typically you might continue from the "core" model. 
        # For demonstration, let's continue from rebuilt model:
        iterative_model = rebuilt_model

    return metrics_pruned, metrics_rebuilt

def main():
    wandb.init(project='alexnet_depGraph', name='AlexNet_Iterative_Prune_Run')
    wandb_logger = WandbLogger(log_model=False)

    device = get_device()
    checkpoint_path = "./checkpoints/best_checkpoint_preTrained.ckpt"

    # Load your DepGraphFineTuner model
    # NOTE: This model must define its "model" attribute inside (like "model.model").
    model = DepGraphFineTuner.load_from_checkpoint(checkpoint_path).to(device)

    # Also keep an original copy for referencing in get_pruned_info calls
    original_model = copy.deepcopy(model)

    # Prepare data
    train_dataloader, val_dataloader, test_dataloader = load_data(
        data_dir='./data',
        batch_size=32,
        val_split=0.2
    )

    # Create the trainer (Lightning) object
    trainer = pl.Trainer(max_epochs=5, logger=wandb_logger, accelerator=device.type)

    # Evaluate model before iterative pruning
    orig_params = count_parameters(model)
    print("MODEL BEFORE PRUNING:\n", model.model)
    print(f"Original number of parameters: {orig_params}")

    orig_accuracy, orig_f1 = evaluate_model(model, test_dataloader, device)
    print(f"Original Accuracy: {orig_accuracy:.4f}, Original F1 Score: {orig_f1:.4f}")
    print("Model size in MB:", model_size_in_mb(model))

    # Perform iterative pruning with 3 steps, each pruning 20%, 30%, 40% for example
    # Or define your own ratios
    pruning_steps = 3
    pruning_ratios = [0.2, 0.3, 0.4]  # just an example
    metrics_pruned, metrics_rebuilt = iterative_pruning(
        model=model,
        original_model=original_model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        pruning_steps=pruning_steps,
        pruning_ratios=pruning_ratios,
        epochs=5,
        learning_rate=1e-4
    )

    print("\n=== Final Results (Pruned) ===")
    print(metrics_pruned)

    print("\n=== Final Results (Rebuilt) ===")
    print(metrics_rebuilt)

    wandb.finish()


if __name__ == "__main__":
    main()