import os
import pytorch_lightning
import torch.utils.data
import torch
import pandas as pd
from utils import train_aug, transform, set_seed
from dataset import Dataset

class DataLoader(pytorch_lightning.LightningDataModule):
    """
    LightningDataModule that handles the creation of train, val, test dataloaders.
    It uses a CSV file that contains split information (train/dev/test).
    """
    def __init__(self, data_version, root_path, batch_size):
        super().__init__()
        self.video_path = f"{root_path}/data/{data_version}.csv"
        self.batch_size = batch_size

        splits = pd.read_csv(self.video_path)
        self.train_data = splits[splits.split == "train"].copy()
        dev_data = splits[splits.split == "dev"].copy()
        test_data = splits[splits.split == "test"].copy()

        self.train_dataset = Dataset(
            data_frame=self.train_data,
            transform=transform,
            augmentation=train_aug,
        )
        self.val_dataset = Dataset(
            data_frame=dev_data,
            transform=transform,
            augmentation=train_aug,
        )
        self.test_dataset = Dataset(
            data_frame=test_data,
            transform=transform,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.RandomSampler(self.train_dataset),
            num_workers=8,
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=True,
            worker_init_fn=set_seed
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SequentialSampler(self.val_dataset),
            num_workers=8,
            prefetch_factor=3,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SequentialSampler(self.test_dataset),
            num_workers=4,
            prefetch_factor=2,
        )

    def get_trainset_size(self):
        return len(self.train_data)