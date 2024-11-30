import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Proceed with imports and downloading models/datasets


import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW, lr_scheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

# Define AlexNet model with CIFAR-10 adaptation
class AlexNetFineTuner(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, num_classes=10):                             #TODO: Try different learning rates: e.g., 1e-3 or 5e-4
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
    wandb.init(project="alexnet_cifar10", name="AlexNet_Training")
    wandb_logger = WandbLogger(log_model=False)

    # Data transformations for CIFAR-10
    transform = Compose([
        Resize((224, 224)),  # Resize images to 224x224 for AlexNet
        ToTensor(),          # Convert images to PyTorch tensors
        Normalize(           # Normalize using ImageNet's mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # # Use a subset for faster training (optional)
    # train_dataset = Subset(train_dataset, range(5000))  # 500 samples for training              #TODO: Change ranges
    # test_dataset = Subset(test_dataset, range(500))    # 100 samples for testing

    # Data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)      #TODO: can change num_workers
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    # Define the model
    model = AlexNetFineTuner()

    # Define the checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpointsAlex/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    # Define the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=5,  # Adjust epochs as needed                                #TODO: Can change epoch
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloaders=test_dataloader)

    # Test the model
    trainer.test(model, dataloaders=test_dataloader)

    # Finish logging
    wandb.finish()

if __name__ == "__main__":
    main()
