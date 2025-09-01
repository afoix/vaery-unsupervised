import logging

import numpy as np
import torch
import zarr
from imageio import imread
from iohub.ngff import ImageArray, Plate, Position, open_ome_zarr
from lightning.pytorch import LightningDataModule
from monai.data import set_track_meta
from monai.data.utils import collate_meta_tensor
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    DictTransform,
    MapTransform,
    MultiSampleTrait,
    RandAffined,
)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

_logger = logging.getLogger("lightning.pytorch")

class DemoDataset(Dataset):
    def __init__(self, positions:list[Position], transform: DictTransform|None=None, train:bool=True
    ):
        super().__init__()
        self.positions = positions
        self.dataset = dataset
        self.transform = transform
        self.train = train
    def __len__(self):
        NotImplementedError("This method is not implemented")
        return len(self.data)

    def __getitem__(self, index):
        # Logic to get a sample from the dataset
        if self.train:
            index = self.train_indices[index]
        else:
            index = self.val_indices[index]

        sample = open_ome_zarr(self.positions[index])

        if self.transform:
            NotImplementedError("This method is not implemented")
            # self.transform a list of transform (crop, rotation, flip, normalize, etc)
            
            sample = self.transform(sample)

        return sample 


class DemoDataModule(LightningDataModule):
    def __init__(self, data_paths:list[str], batch_size:int, num_workers:int,prefetch_factor:int=2,pin_memory:bool=True,persistent_workers:bool=False):
        super().__init__()
        self.data_paths = data_paths

        self.dataset = None
        self.batch_size = batch_size
        self.num_workers = num_workers

        # NOTE: These parameters are for performance
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
    
    def prepare_data(self):
        NotImplementedError("This method is not implemented")

    def setup(self, stage: str):
        if stage == "fit" or stage == "train":
            self.dataset = DemoDataset(self.data_paths)
            NotImplementedError("This method is not implemented")
        elif stage == "val" or stage == "validate":
            self.dataset = DemoDataset(self.data_paths, train=False)
            NotImplementedError("This method is not implemented")
        elif stage == "predict":
            NotImplementedError("This method is not implemented")

    def train_dataloader(self):
        return DataLoader(self.dataset,shuffle=True,batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True,prefetch_factor=self.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(self.dataset,shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True,prefetch_factor=self.prefetch_factor)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True,prefetch_factor=self.prefetch_factor)