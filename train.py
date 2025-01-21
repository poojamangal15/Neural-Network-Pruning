import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from utils.data_utils import load_data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from models.alexNet_fineTuner import AlexNetFineTuner

def load_cifar10_with_alexnet_transforms(data_dir, batch_size, val_split=0.2):
    from torchvision import datasets, transforms
    from torch.utils.data import random_split, DataLoader

    # Transform with resizing and normalization for ImageNet-pretrained AlexNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)

    # Split train dataset into train and validation sets
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def main():
    # Initialize Wandb
    wandb.init(project="alexnet_cifar10", name="Pretrained_alexNet_epoch200")
    wandb_logger = WandbLogger(log_model=False)

    # Load CIFAR-10 dataset
    train_loader, val_loader, test_loader = load_cifar10_with_alexnet_transforms(
        data_dir='./data', batch_size=32, val_split=0.2
    )

    # Initialize the model
    model = AlexNetFineTuner(learning_rate=1e-5, num_classes=10, freeze_features=False)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best_checkpoint',
        save_top_k=1,
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=200,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, dataloaders=test_loader)

    # Finish Wandb
    wandb.finish()

if __name__ == "__main__":
    main()
