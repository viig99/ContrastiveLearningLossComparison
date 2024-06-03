import torch
import torchvision
import pytorch_lightning as pl
from torch import nn
from lightly.models.modules import SimCLRProjectionHead
from lib.losses import InfoNCELoss
from transformers import get_cosine_schedule_with_warmup


class SimCLR(pl.LightningModule):
    def __init__(self, total_steps: int):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 2048, 2048)
        self.criterion = InfoNCELoss()
        self.total_steps = total_steps
        self.warmup_steps = int(0.05 * total_steps)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("batch_ce_loss", loss, prog_bar=True, on_step=True)
        self.log("epoch_ce_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            eps=1e-6,
            fused=True,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optim,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        return [optim], [{"scheduler": scheduler, "interval": "step"}]


class LinearClassifier(pl.LightningModule):
    def __init__(
        self, backbone_model: pl.LightningModule, num_classes: int, total_steps: int
    ):
        super().__init__()
        self.backbone = backbone_model
        self.fc = nn.Linear(2048, num_classes)
        self.total_steps = total_steps
        self.warmup_steps = int(0.05 * total_steps)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        return self.fc(x)

    def training_step(self, batch, batch_index):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=True)

        # calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_index):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("test_loss", loss, prog_bar=True)

        # calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            eps=1e-6,
            fused=True,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optim,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        return [optim], [{"scheduler": scheduler, "interval": "step"}]
