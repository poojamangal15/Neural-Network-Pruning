from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose, RandomResizedCrop, RandomHorizontalFlip, RandomRotation,
    ColorJitter, ToTensor, Normalize, Resize, CenterCrop
)
from torch.utils.data import random_split, DataLoader
import torch

def load_data(data_dir='./data', batch_size=32, val_split=0.2):
    # Data transformations for training
    train_transform = Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Data transformations for validation/test
    test_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # Split the train dataset into train/val
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader


# def load_imagenette_data(
#     data_dir: str = './data/imagenette2',
#     batch_size: int = 32,
#     val_split: float = 0.2,
#     num_workers: int = 4
# ):
#     """
#     Loads Imagenette2-160 from:
#         data_dir/train/<class>/*.JPEG
#         data_dir/val/<class>/*.JPEG

#     Splits the train folder into train/val, and uses val folder as test set.
#     Returns: train_loader, val_loader, test_loader
#     """
#     # --- Transforms ---
#     train_transform = Compose([
#         RandomResizedCrop(160),
#         RandomHorizontalFlip(p=0.5),
#         RandomRotation(degrees=15),
#         ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         ToTensor(),
#         Normalize(mean=[0.485, 0.456, 0.406],
#                   std=[0.229, 0.224, 0.225]),
#     ])
#     test_transform = Compose([
#         Resize(160),
#         CenterCrop(160),
#         ToTensor(),
#         Normalize(mean=[0.485, 0.456, 0.406],
#                   std=[0.229, 0.224, 0.225]),
#     ])

#     # --- Datasets ---
#     full_train = ImageFolder(root=f"{data_dir}/train", transform=train_transform)
#     test_set  = ImageFolder(root=f"{data_dir}/val",   transform=test_transform)

#     # split train→ train/val
#     val_size   = int(len(full_train) * val_split)
#     train_size = len(full_train) - val_size
#     train_set, val_set = random_split(full_train, [train_size, val_size])

#     # --- Dataloaders ---
#     train_loader = DataLoader(
#         train_set, batch_size=batch_size, shuffle=True,
#         num_workers=num_workers, pin_memory=True
#     )
#     val_loader = DataLoader(
#         val_set, batch_size=batch_size, shuffle=False,
#         num_workers=num_workers, pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_set, batch_size=batch_size, shuffle=False,
#         num_workers=num_workers, pin_memory=True
#     )

#     return train_loader, val_loader, test_loader

# data_utils.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_imagenette(data_dir="./data/imagenette2",
                    batch_size=128,
                    val_split=0.1,
                    num_workers=8):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.Resize(40),          # small augmentation
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])

    full_train = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
    n_val = int(len(full_train) * val_split)
    train_ds, val_ds = random_split(full_train, [len(full_train)-n_val, n_val])
    test_ds = datasets.ImageFolder(f"{data_dir}/val", transform=val_tf)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)