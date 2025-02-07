from torchvision import models
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
import torch

class AlexNetFineTuner(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, num_classes=10, freeze_features=False):
        super(AlexNetFineTuner, self).__init__()
        self.save_hyperparameters()

        # Load ImageNet-pretrained AlexNet
        self.model = models.alexnet(pretrained=True)

        # Modify the classifier for CIFAR-10 (10 classes)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

        # Optionally freeze feature extractor
        if freeze_features:
            for param in self.model.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        val_loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        val_acc = (preds == labels).float().mean()
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        test_loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        test_acc = (preds == labels).float().mean()
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", test_acc, prog_bar=True)
        return test_loss

    def on_test_epoch_end(self):
        avg_loss = sum(o["test_loss"] for o in self.test_outputs) / len(self.test_outputs)
        avg_acc = sum(o["test_acc"] for o in self.test_outputs) / len(self.test_outputs)
        self.log("test_loss_epoch", avg_loss, prog_bar=True)
        self.log("test_acc_epoch", avg_acc, prog_bar=True)
        self.test_outputs = []

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


   
