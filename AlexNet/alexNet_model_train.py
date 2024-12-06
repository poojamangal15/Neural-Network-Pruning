import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Proceed with imports and downloading models/datasets


import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import random_split, DataLoader, Subset
from torch.optim import AdamW, lr_scheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter, RandomCrop
from pytorch_lightning.callbacks import EarlyStopping

import wandb


# Define AlexNet model with CIFAR-10 dataset
class AlexNetFineTuner(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, num_classes=10):                             #TODO: Try different learning rates: e.g., 1e-3 or 5e-4
        super(AlexNetFineTuner, self).__init__()
        self.save_hyperparameters()
        
        # Load pre-trained AlexNet model
        self.model = models.alexnet(pretrained=True)
        
        # Replace the final classifier layer to match CIFAR-10's 10 classes
        self.model.classifier[6] = torch.nn.Linear(4096, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

def main():
    # Initialize Weights & Biases logging
    wandb.init(project="alexnet_cifar10", name="AlexNet_with_aug_1e-5_epoch5")
    wandb_logger = WandbLogger(log_model=False)

    # Data transformations for CIFAR-10

    train_transform = Compose([
        Resize((224, 224)),                  # Resize images to 224x224
        RandomHorizontalFlip(p=0.5),        # Randomly flip the image horizontally with 50% probability
        RandomRotation(degrees=15),         # Rotate the image randomly within 15 degrees
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color adjustments
        ToTensor(),                         # Convert images to PyTorch tensors
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    # Normalize using ImageNet's mean and std
    ])
 
    test_transform = Compose([
        Resize((224, 224)),  # Resize images to 224x224 for AlexNet
        ToTensor(),          # Convert images to PyTorch tensors
        Normalize(           # Normalize using ImageNet's mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # # Use a subset for faster training (optional)
    # train_dataset = Subset(train_dataset, range(5000))  # 500 samples for training              #TODO: Change ranges
    # test_dataset = Subset(test_dataset, range(500))    # 100 samples for testing

    train_size = int(0.8 * len(train_dataset))  # 80% training
    val_size = len(train_dataset) - train_size  # 20% validation
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    # Define the model
    model = AlexNetFineTuner()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,  # Number of epochs to wait before stopping
        mode='min',
        verbose=True  
    )
    # Define the checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpointsAlex/',
        filename='checkpoint_with_aug_epoch5',
        save_top_k=1,
        mode='min'
    )

    # Define the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=10,  # Adjust epochs as needed                                #TODO: Can change epoch
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

    # Test the model
    trainer.test(model, dataloaders=test_dataloader)

    # Finish logging
    wandb.finish()

if __name__ == "__main__":
    main()
