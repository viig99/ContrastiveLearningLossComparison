import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE

# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR100
from torchvision import transforms


class CIFAR100DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./datasets",
        num_workers: int = 8,
        batch_size: int = 32,
        batch_factor: int = 2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = SimCLRTransform()
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]
                ),
            ]
        )
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.batch_factor = batch_factor

    def prepare_data(self):
        # download
        CIFAR100(self.data_dir, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "pretrain":
            self.cifar_pretrain = CIFAR100(
                self.data_dir, train=True, transform=self.transform
            )

        if stage == "finetune":
            cifar_train_full = CIFAR100(
                self.data_dir, train=True, transform=self.val_transforms
            )
            generator = torch.Generator().manual_seed(42)
            self.cifar_finetune_train, self.cifar_finetune_val = random_split(
                cifar_train_full, [0.9, 0.1], generator=generator
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.cifar_test = CIFAR100(
                self.data_dir, train=False, transform=self.val_transforms
            )

    def pretrain_dataloader(self):
        return DataLoader(
            self.cifar_pretrain,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_finetune_train,
            batch_size=self.batch_size * self.batch_factor,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_finetune_val,
            batch_size=self.batch_size * self.batch_factor,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size * self.batch_factor,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
