# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
import os
import pytorch_lightning as pl

torch.set_float32_matmul_precision("medium")

from lib.model import SimCLR
from lib.data import CIFAR100DataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from glob import glob


class Config:
    num_workers = 10
    batch_size = 128
    max_epochs = 200
    precision = "bf16-true"
    log_every_n_steps = 10
    checkpoint_dir = "checkpoints"


def get_total_steps(dataloader, batch_size, epochs):
    return len(dataloader) * epochs


def get_last_checkpoint(checkpoint_dir: str):
    checkpoints = glob(f"{checkpoint_dir}/*.ckpt")

    if not checkpoints:
        return None

    # Get the last saved checkpoint based on creation time
    return max(checkpoints, key=os.path.getctime)


if __name__ == "__main__":
    cfg = Config()

    datamodule = CIFAR100DataModule(
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    pretrain_data = datamodule.pretrain_dataloader()

    model = SimCLR(
        total_steps=get_total_steps(pretrain_data, cfg.batch_size, cfg.max_epochs)
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback_pretrain = ModelCheckpoint(
        monitor="epoch_ce_loss",
        dirpath=cfg.checkpoint_dir,
        filename="simclr-cifar100-{epoch:02d}-{epoch_ce_loss:.2f}",
        save_top_k=2,
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        strategy="auto",
        precision=cfg.precision,  # type: ignore
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=[lr_monitor, checkpoint_callback_pretrain],
        benchmark=False,
        deterministic=True,
    )

    trainer.fit(
        model=model,
        train_dataloaders=pretrain_data,
        ckpt_path=get_last_checkpoint(cfg.checkpoint_dir),
    )

    # Fine-tuning phase
    # finetune_data = datamodule.train_dataloader()
    # finetune_val_data = datamodule.val_dataloader()

    # final_test_data = datamodule.test_dataloader()
