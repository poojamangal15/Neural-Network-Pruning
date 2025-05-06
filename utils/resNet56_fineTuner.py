import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
import torch.hub

class ResNet56FineTuner(pl.LightningModule):
    """
    CIFAR‑10 ResNet‑56 → fine‑tune on Imagenette (10 classes, 32×32 inputs).
    No architectural surgery.
    """
    def __init__(self, lr=1e-3, num_classes=10, freeze_features=False):
        super().__init__()
        self.save_hyperparameters()

        # 1) load backbone (pretrained on CIFAR‑10)
        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_resnet56",
            pretrained=True
        )
        # 2) swap classifier
        self.model.fc = torch.nn.Linear(64, num_classes)

        # 3) optional freeze conv body
        if freeze_features:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.fc.parameters():
                p.requires_grad = True    # keep head trainable

    # forward
    def forward(self, x): return self.model(x)

    # shared train/val/test step
    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc  = (logits.argmax(1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=(stage!="train"))
        self.log(f"{stage}_acc",  acc,  prog_bar=True)
        return loss

    def training_step  (self, b, _): return self._step(b, "train")
    def validation_step(self, b, _): return self._step(b, "val")
    def test_step      (self, b, _): return self._step(b, "test")

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sched = lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=3)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
