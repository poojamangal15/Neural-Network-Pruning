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

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DistilBERTFineTuner(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, num_labels=2):
        super(DistilBERTFineTuner, self).__init__()
        self.save_hyperparameters()
        
        # Initialize DistilBERT model and tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

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

    def test_step(self, batch, batch_idx):
        inputs = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Forward pass through self.model
        outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        test_loss = outputs.loss  # Test loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        # Calculate accuracy for multiclass classification
        test_acc = accuracy(preds, labels, task="multiclass", num_classes=4)
        
        # Log test loss and accuracy
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", test_acc, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


def main():
    # Prepare dataset with a smaller subset
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    def tokenize_data(batch):
        tokens = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=64)
        tokens["labels"] = batch["label"]  
        return tokens

    # Load and shuffle the ag_news dataset
    dataset = load_dataset("ag_news").shuffle(seed=42)

    # Select only a subset for quick training
    dataset["train"] = dataset["train"].select(range(500))  # Select 500 samples for training
    dataset["test"] = dataset["test"].select(range(100))    # Select 100 samples for testing

    # Tokenize the dataset
    dataset = dataset.map(tokenize_data, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])  

    # Create data loaders for training and validation
    train_dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset["test"], batch_size=32)

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',                 # Monitor validation loss
        dirpath='checkpoints/',              # Directory to save the checkpoints
        filename='best-checkpoint',          # Filename for the best checkpoint
        save_top_k=1,                        # Only keep the best model
        mode='min'                           # Save the model with the minimum val_loss
    )

    # Initialize model, logger, and trainer
    wandb.init(project='distilbert_pruning', name='Test_Run')
    wandb_logger = WandbLogger(log_model=False)
    model = DistilBERTFineTuner()

    trainer = pl.Trainer(
        max_epochs=3,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]      # Add checkpoint callback to the trainer
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloaders=test_dataloader)

    # Evaluate the model on the test set
    trainer.test(model, dataloaders=test_dataloader)

    # Finish logging
    wandb.finish()


# Only run main if this script is executed directly
if __name__ == "__main__":
    main()
