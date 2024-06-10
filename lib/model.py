import torch
import torchvision
import pytorch_lightning as pl
from torch import nn
from lightly.models.modules import SimCLRProjectionHead
from transformers import get_cosine_schedule_with_warmup
from enum import Enum
from lib.losses import *


class Losses(Enum):
    info_nce = InfoNCELoss
    dcl = DCL
    dcl_symmetric = DCL_symmetric
    nt_xent = NT_xent
    dhel = DHEL
    vicreg = VICReg

    @staticmethod
    def get_choices():
        return [loss.name for loss in Losses]


class BatchTimer:

    def __init__(self, logger):
        self.logger = logger

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()  # type: ignore
        return self

    def __exit__(self, *args):
        self.end.record()  # type: ignore
        torch.cuda.synchronize()
        self.elapsed_time = self.start.elapsed_time(self.end)
        self.logger("batch_time_ms", self.elapsed_time, prog_bar=True, on_step=True)


def configure_optimizers(
    named_parameters, warmup_steps, total_steps, weight_decay=None
):
    if weight_decay and weight_decay > 0:
        param_dict = {pn: p for pn, p in named_parameters if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() > 1]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() <= 1]

        print(
            f"Decay tensors - params: {len(decay_params)} - {sum(p.numel() for p in decay_params)}"
        )
        print(
            f"No Decay tensors - params: {len(no_decay_params)} - {sum(p.numel() for p in no_decay_params)}"
        )

        optim_groups = [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
    else:
        get_params_from = lambda np: (p for _, p in np)
        optim_groups = get_params_from(named_parameters)

    optim = torch.optim.AdamW(
        optim_groups,
        lr=1e-3,
        eps=1e-8,
        betas=(0.9, 0.95),
        fused=True,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.3,
    )

    return optim, scheduler


class SimCLR(pl.LightningModule):
    def __init__(self, total_steps: int, temperature: float, loss_func_name: str):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 2048, 2048, num_layers=3)
        loss_func = getattr(Losses, loss_func_name).value
        self.criterion = loss_func(temperature)
        self.total_steps = total_steps
        self.warmup_steps = int(0.05 * total_steps)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        with BatchTimer(self.log):
            z0 = self.forward(x0)
            z1 = self.forward(x1)
            loss = self.criterion(z0, z1)
        self.log("batch_ce_loss", loss, prog_bar=True, on_step=True)
        self.log("epoch_ce_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim, scheduler = configure_optimizers(
            self.named_parameters(),
            self.warmup_steps,
            self.total_steps,
            weight_decay=None,
        )
        return [optim], [{"scheduler": scheduler, "interval": "step"}]


class LinearClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone_model: pl.LightningModule,
        num_classes: int,
        total_steps: int,
        fully_finetune: bool = False,
    ):
        super().__init__()
        self.backbone = backbone_model
        self.fc = nn.Linear(2048, num_classes)
        self.total_steps = total_steps
        self.warmup_steps = int(0.05 * total_steps)
        self.loss = nn.CrossEntropyLoss()
        self.fully_finetune = fully_finetune

    def forward(self, x):
        if not self.fully_finetune:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        return self.fc(x)

    def training_step(self, batch, batch_index):
        x, y = batch
        with BatchTimer(self.log):
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
        acc_top_1 = (preds == y).float().mean()
        acc_top_5 = (
            torch.topk(logits, k=5, dim=1)
            .indices.eq(y.unsqueeze(1))
            .sum(dim=1)
            .float()
            .mean()
        )
        self.log(
            "val_acc_top_1", acc_top_1, prog_bar=True, on_epoch=True, on_step=False
        )
        self.log(
            "val_acc_top_5", acc_top_5, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def test_step(self, batch, batch_index):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("test_loss", loss, prog_bar=True)

        # calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc_top_1 = (preds == y).float().mean()
        acc_top_5 = (
            torch.topk(logits, k=5, dim=1)
            .indices.eq(y.unsqueeze(1))
            .sum(dim=1)
            .float()
            .mean()
        )
        self.log(
            "test_acc_top_1", acc_top_1, prog_bar=True, on_epoch=True, on_step=False
        )
        self.log(
            "test_acc_top_5", acc_top_5, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def configure_optimizers(self):
        optim, scheduler = configure_optimizers(
            self.named_parameters(),
            self.warmup_steps,
            self.total_steps,
            weight_decay=None,
        )
        return [optim], [{"scheduler": scheduler, "interval": "step"}]
