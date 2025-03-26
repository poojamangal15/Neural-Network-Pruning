from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter
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


from torchvision import datasets, transforms

def load_data_imagenet(data_dir='/path/to/imagenet', batch_size=256, workers=8):
    """
    Loads ImageNet data from a standard folder structure:
      data_dir/train/<class>/*.JPEG
      data_dir/val/<class>/*.JPEG

    Returns:
        train_dataloader, val_dataloader
    """

    # Standard ImageNet normalization
    # (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Typical transforms:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # The standard subfolders: data_dir/train & data_dir/val
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=f"{data_dir}/val",   transform=val_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )

    return train_dataloader, val_dataloader
