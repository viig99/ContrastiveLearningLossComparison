import torch
import pytorch_lightning as pl
import argparse

torch.set_float32_matmul_precision("medium")

from lib.model import LinearClassifier
from lib.data import CIFAR100DataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision.models import resnet18


from train import Config, get_total_steps


class resnet_backbone(torch.nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        resnet = resnet18()
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.expand_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, num_features),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.expand_features(x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--batch_factor", type=int, default=Config.batch_factor)
    parser.add_argument("--num_workers", type=int, default=Config.num_workers)
    args = parser.parse_args()

    cfg = Config(args)

    datamodule = CIFAR100DataModule(
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        batch_factor=cfg.batch_factor,
    )
    datamodule.prepare_data()

    datamodule.setup(stage="finetune")

    resnet_model = resnet18()

    backbone_model = resnet_backbone(2048)

    finetune_data = datamodule.train_dataloader()
    finetune_val_data = datamodule.val_dataloader()

    num_epochs = cfg.finetune_epochs
    batch_size = cfg.batch_size * cfg.batch_factor

    finetune_model = LinearClassifier(
        backbone_model=backbone_model,  # type: ignore
        num_classes=cfg.num_classes,
        total_steps=get_total_steps(finetune_data, batch_size, num_epochs),
        fully_finetune=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    finetune_trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        strategy="auto",
        precision=cfg.precision,  # type: ignore
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=[lr_monitor],
        benchmark=False,
        deterministic=True,
    )
    finetune_trainer.fit(
        model=finetune_model,
        train_dataloaders=finetune_data,
        val_dataloaders=finetune_val_data,
    )

    datamodule.setup("test")
    finetune_trainer.test(
        model=finetune_model, dataloaders=datamodule.test_dataloader()
    )
