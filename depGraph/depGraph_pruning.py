import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
import pytorch_lightning as pl
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import load_dataset
import wandb
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import torch_pruning as tp

import networkx as nx
import matplotlib.pyplot as plt
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DistilBERTFineTuner(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, num_labels=2):
        super(DistilBERTFineTuner, self).__init__()
        self.save_hyperparameters()

        # Initialize DistilBERT model and tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

        # Initialize metrics dictionary
        self.metrics = {
            "pruning_percentage": [],
            "test_accuracy": [],
            "test_loss": [],
            "model_size": []
        }

    def print_dependency_graph(self, DG):
        """
        Print the dependency graph details for debugging.

        Args:
        - DG (torch_pruning.DependencyGraph): The Dependency Graph object.
        """
        print("\nDependency Graph Details:")
        for module, node in DG.module2node.items():
            print(f"Module: {module}")
            print(f"  - Dependencies:")
            for dep in node.dependencies:  # Use 'dependencies' instead of 'dependency_sets'
                print(f"    * Target Module: {dep.target.module}")
                print(f"      Dependency Info: {vars(dep)}")  # Print available attributes of the dependency




    def visualize_dependency_graph(self, DG):
        """
        Visualize the dependency graph using networkx.

        Args:
        - DG (torch_pruning.DependencyGraph): The Dependency Graph object.
        """
        G = nx.DiGraph()

        # Add edges for each dependency
        for module, node in DG.module2node.items():
            for dep in node.dependencies:
                G.add_edge(str(module), str(dep.target.module))  # Use str(module) to simplify names

        # Plot the graph
        plt.figure(figsize=(12, 8))
        nx.draw(G, with_labels=True, node_size=1000, font_size=8, node_color="skyblue", edge_color="gray")
        plt.title("Dependency Graph")
        plt.show()

    def prune_model(self, pruning_percentage=0.2):
        example_inputs = {
            "input_ids": torch.randint(0, 100, (1, 64)),
            "attention_mask": torch.ones(1, 64),
        }

        # Build Dependency Graph
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs)
        self.print_dependency_graph(DG)
        self.visualize_dependency_graph(DG)


        layer_to_prune = self.model.pre_classifier

        # Log model size before pruning
        self.metrics["model_size"].append(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.metrics["pruning_percentage"].append(0 if len(self.metrics["pruning_percentage"]) == 0 else pruning_percentage * 100)

        # Pruning
        pruning_idxs = list(range(int(layer_to_prune.out_features * pruning_percentage)))
        group = DG.get_pruning_group(layer_to_prune, tp.prune_linear_out_channels, idxs=pruning_idxs)
        if DG.check_pruning_group(group):
            group.prune()

        # Log model size after pruning
        self.metrics["model_size"].append(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.metrics["pruning_percentage"].append(pruning_percentage * 100)

    def test_step(self, batch, batch_idx):
        """
        Override the test step to log accuracy and loss dynamically.
        """
        inputs = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        test_loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        test_acc = accuracy(preds, labels, task="multiclass", num_classes=4)

        # Save metrics
        self.metrics["test_accuracy"].append(test_acc.item())
        self.metrics["test_loss"].append(test_loss.item())

        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", test_acc, prog_bar=True)
        return test_loss

    


    def training_step(self, batch, batch_idx):
        inputs = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Forward pass through self.model
        outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # Get the loss from the model outputs
        
        # Log the loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Forward pass through self.model
        outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        val_loss = outputs.loss  # Validation loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        # Calculate accuracy for multiclass classification
        val_acc = accuracy(preds, labels, task="multiclass", num_classes=4)
        
        # Log validation loss and accuracy
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        return val_loss


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


def plot_metrics(metrics):
        """
        Generate graphs dynamically based on collected metrics.
        
        Args:
        - metrics (dict): Dictionary with keys for pruning percentages, accuracy, loss, and model size.
        """
        import matplotlib.pyplot as plt

        # Pruning Percentage vs. Test Accuracy
        plt.figure()
        plt.plot(metrics["pruning_percentage"], metrics["test_accuracy"], marker='o', label="Accuracy")
        plt.title("Test Accuracy vs. Pruning Percentage")
        plt.xlabel("Pruning Percentage (%)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Pruning Percentage vs. Test Loss
        plt.figure()
        plt.plot(metrics["pruning_percentage"], metrics["test_loss"], marker='o', color="orange", label="Loss")
        plt.title("Test Loss vs. Pruning Percentage")
        plt.xlabel("Pruning Percentage (%)")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Pruning Percentage vs. Model Size
        plt.figure()
        plt.plot(metrics["pruning_percentage"], metrics["model_size"], marker='o', color="green", label="Model Size")
        plt.title("Model Size vs. Pruning Percentage")
        plt.xlabel("Pruning Percentage (%)")
        plt.ylabel("Number of Parameters")
        plt.grid(True)
        plt.legend()
        plt.show()

def main():
    # Prepare dataset with a smaller subset
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_data(batch):
        tokens = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=64)
        tokens["labels"] = batch["label"]
        return tokens

    dataset = load_dataset("ag_news").shuffle(seed=42)
    dataset["train"] = dataset["train"].select(range(500))
    dataset["test"] = dataset["test"].select(range(100))
    dataset = dataset.map(tokenize_data, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset["test"], batch_size=32)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    wandb.init(project='distilbert_pruning', name='Test_Run')
    wandb_logger = WandbLogger(log_model=False)
    model = DistilBERTFineTuner()

    trainer = pl.Trainer(
        max_epochs=3,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_dataloader, val_dataloaders=test_dataloader)
    model.prune_model(pruning_percentage=0.1)

    # Apply pruning and test
    # for pruning_percentage in [0.1, 0.2, 0.3]:  # Test different pruning percentages
    #     print(f"Applying {pruning_percentage * 100}% pruning...")
    #     model.prune_model(pruning_percentage=pruning_percentage)
    #     trainer.test(model, dataloaders=test_dataloader)

    # Plot metrics
    # plot_metrics(model.metrics)

    wandb.finish()


if __name__ == "__main__":
    main()
