import os, pytorch_lightning as pl, wandb

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from utils.data_utils import load_imagenette
from utils.resNet56_fineTuner import ResNet56FineTuner

DATA_DIR   = "./data/imagenette2"
BATCH      = 128
EPOCHS     = 100
LR         = 1e-3

def main():
    # data
    train_loader, val_loader, test_loader = load_imagenette(DATA_DIR, BATCH)

    # model
    model = ResNet56FineTuner(lr=LR, num_classes=10)

    # callbacks
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="res56_imagenette",
        monitor="val_loss", mode="min", save_top_k=1, auto_insert_metric_name=False
    )
    es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=10)

    # logger
    wandb_logger = WandbLogger(project="ResNet56-Imagenette-32px")

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=EPOCHS,
        callbacks=[ckpt_cb, es_cb],
        logger=wandb_logger,
        log_every_n_steps=50
    )

    trainer.fit(model, train_loader, val_loader)
    test_metrics = trainer.test(ckpt_path=ckpt_cb.best_model_path,
                                dataloaders=test_loader)
    print(f"✅  Top‑1 test accuracy: {test_metrics[0]['test_acc']:.4f}")
    print("Checkpoint:", ckpt_cb.best_model_path)
    wandb.finish()

if __name__ == "__main__":
    main()
