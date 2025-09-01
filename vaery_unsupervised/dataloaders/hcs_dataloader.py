from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
from iohub import open_ome_zarr
from iohub.ngff import Position
from monai.data import set_track_meta
from monai.transforms import Compose, ToTensord
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class ContrastiveHCSDataset(Dataset):
    def __init__(self, positions: list[Position], source_channel_names: list[str], crop_size: tuple[int, int]=(256, 256), 
                 crops_per_position: int=4, normalization_transform: Compose| list[Callable]=None, anchor_augmentations: Callable=None, positive_augmentations: Callable=None):
        """
        Parameters:
            positions: List of iohub.ngff.Position objects
            source_channel_names: Channel names to use
            crop_size: Size of random crops
            crops_per_position: Number of crops to generate per position per epoch
            normalization_transform: MONAI transform for normalization (optional)
            anchor_augmentations: MONAI transform for anchor augmentations (optional)
            positive_augmentations: MONAI transform for positive augmentations (optional)
        """
        self.positions = positions
        self.source_channel_names = source_channel_names
        self.crop_size = crop_size
        self.crops_per_position = crops_per_position
        self.normalization_transform = normalization_transform
        self.anchor_augmentations = anchor_augmentations
        self.positive_augmentations = positive_augmentations
            
    def __getitem__(self, idx):

        #TODO open the 
        # 
        position = self.positions[idx]
        #TODO open the position
        img_tczyx =position.data[:]

        #TODO apply the transform and the normalizations
        anchor = self.anchor_augmentations(img_tczyx)
        positive = self.positive_augmentations(img_tczyx)


        #TODO return the anchor and positive
        return {
            "anchor": anchor,
            "positive": positive,
            # "fov_id": position.name #TODO check if we need the fov_id
        }
    
    def __len__(self):
        return len(self.positions) * self.crops_per_position


class HCSDataModule(pl.LightningDataModule):
    def __init__(self, ome_zarr_path, source_channel_names, crop_size=(256, 256),
                 crops_per_position=4, batch_size=32, num_workers=4, 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42, 
                 normalization_transform=None, augmentations=None):
        super().__init__()
        self.ome_zarr_path = ome_zarr_path
        self.source_channel_names = source_channel_names
        self.crop_size = crop_size
        self.crops_per_position = crops_per_position
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.normalization_transform = normalization_transform
        self.augmentations = augmentations

    def _set_fit_global_state(self, num_positions: int) -> torch.Tensor:
        # disable metadata tracking in MONAI for performance
        set_track_meta(False)
        # shuffle positions, randomness is handled globally
        return torch.randperm(num_positions)

    def setup(self, stage=None):
        # TODO:
        plate = open_ome_zarr(self.ome_zarr_path, mode="r")
        positions = [pos for _, pos in plate.positions()]
        shuffled_indices = self._set_fit_global_state(len(positions))
        positions = list(positions[i] for i in shuffled_indices)
        
        num_train_fovs = int(len(positions) * self.split_ratio)

        #TODO define what are the train_transform and val_transform
        #TODO ContrastiveHCSDataset extra arguments that are needed
        #TODO check that we need how many crops per positions
        self.train_dataset = ContrastiveHCSDataset(
            positions[:num_train_fovs],
            transform=train_transform,
            crops_per_position=self.crops_per_position,
            **train_dataset_settings,
        )
        self.val_dataset = ContrastiveHCSDataset(
            positions[num_train_fovs:],
            crops_per_position=self.crops_per_position,
            transform=val_transform,
            **dataset_settings,
        )
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         shuffle=False, num_workers=self.num_workers)