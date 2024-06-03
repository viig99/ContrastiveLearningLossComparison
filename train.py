# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
import os
import pytorch_lightning as pl

torch.set_float32_matmul_precision("medium")

from lib.model import SimCLR, LinearClassifier
from lib.data import CIFAR100DataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from glob import glob


class Config:
    num_workers = 10
    batch_size = 128
    batch_factor = 4
    max_epochs = 200
    num_classes = 100
    finetune_epochs = 10
    precision = "bf16-true"
    log_every_n_steps = 10
    pretrain_checkpoint_dir = "checkpoints/pretrain"
    finetune_checkpoint_dir = "checkpoints/finetune"
    pretrain = False


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
    datamodule.setup("test")

    if cfg.pretrain:

        pretrain_data = datamodule.pretrain_dataloader()

        model = SimCLR(
            total_steps=get_total_steps(pretrain_data, cfg.batch_size, cfg.max_epochs)
        )

        # Pre-training phase
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callback_pretrain = ModelCheckpoint(
            monitor="epoch_ce_loss",
            dirpath=cfg.pretrain_checkpoint_dir,
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
            ckpt_path=get_last_checkpoint(cfg.pretrain_checkpoint_dir),
        )

    else:

        # Fine-tuning phase
        checkpoint = get_last_checkpoint(cfg.pretrain_checkpoint_dir)

        if checkpoint is None:
            raise ValueError("No checkpoint found")

        backbone_model = SimCLR.load_from_checkpoint(checkpoint, total_steps=0)

        finetune_data = datamodule.train_dataloader()
        finetune_val_data = datamodule.val_dataloader()

        num_epochs = cfg.finetune_epochs
        batch_size = cfg.batch_size * cfg.batch_factor

        finetune_model = LinearClassifier(
            backbone_model=backbone_model,
            num_classes=cfg.num_classes,
            total_steps=get_total_steps(finetune_data, batch_size, num_epochs),
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callback_finetune = ModelCheckpoint(
            monitor="val_acc",
            dirpath=cfg.finetune_checkpoint_dir,
            filename="linear-cifar100-{epoch:02d}-{val_acc:.2f}",
            save_top_k=2,
            mode="max",
            verbose=True,
        )

        finetune_trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="auto",
            strategy="auto",
            precision=cfg.precision,  # type: ignore
            log_every_n_steps=cfg.log_every_n_steps,
            callbacks=[lr_monitor, checkpoint_callback_finetune],
            benchmark=False,
            deterministic=True,
        )
        finetune_trainer.fit(
            model=finetune_model,
            train_dataloaders=finetune_data,
            val_dataloaders=finetune_val_data,
            ckpt_path=get_last_checkpoint(cfg.finetune_checkpoint_dir),
        )

        finetune_trainer.test(
            model=finetune_model, dataloaders=datamodule.test_dataloader()
        )
