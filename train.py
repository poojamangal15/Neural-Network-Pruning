import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb

from models.alexNet_fineTuner import AlexNetFineTuner
from utils.data_utils import load_data

def main():
    # Initialize Weights & Biases logging
    wandb.init(project="alexnet_cifar10", name="AlexNet_with_aug_1e-5_epoch5")
    wandb_logger = WandbLogger(log_model=False)

    # Load the CIFAR-10 dataset
    train_dataloader, val_dataloader, test_dataloader = load_data(data_dir='./data', batch_size=32, val_split=0.2)

    # Define the model
    model = AlexNetFineTuner(learning_rate=1e-5, num_classes=10)

    # Early stopping and checkpointing
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        verbose=True
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best_checkpoint',
        save_top_k=1,
        mode='min'
    )

    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Test the model
    trainer.test(model, dataloaders=test_dataloader)

    # Finish logging
    wandb.finish()

if __name__ == "__main__":
    main()
