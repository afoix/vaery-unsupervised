#%%
from pathlib import Path
import torch
from torch.utils.data import Dataset
from iohub.ngff import open_ome_zarr
import numpy as np
from monai.transforms import Compose, RandRotate, RandSpatialCrop

class PositivePairDataset(Dataset):
    def __init__(self, zarr_path, samples, patch_size=(384,384), transforms=None):
        """
        Args:
            zarr_path (str or Path): path to your .zarr plate
            samples (list of tuples): list of positive pairs of positions
                each element: ((row, col, field), (row, col, field))
            patch_size (tuple): final YX patch size
            transforms (monai.transforms.Compose, optional): optional augmentation pipeline
        """
        self.zarr_path = Path(zarr_path)
        self.samples = samples
        self.patch_size = patch_size
        # self.roi_size = roi_size
        
        if transforms is None:
            self.transforms = Compose([
                RandSpatialCrop(roi_size = patch_size, max_roi_size=None, random_center=True, random_size=False, lazy=False),
                RandRotate(range_x=30, prob=0.5, keep_size=True, mode="bilinear"),
                RandSpatialCrop(roi_size = (256,256), max_roi_size=None, random_center=True, random_size=False, lazy=False),
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def _load_image(self, position):
        row, col, field = position
        with open_ome_zarr(self.zarr_path / str(row) / str(col) / str(field), mode="r") as store:
            arr = store["0"]  # shape (t=1, c=3, z=1, y, x)
            image = arr[0, :, 0, :, :]  # select t=0, all channels, z=0
            return torch.tensor(image, dtype=torch.float32)
        

    def __getitem__(self, idx):
        position= self.samples[idx]
        anchor = self._load_image(position)


        pos_pair = self.transforms(anchor)

      
        return {"anchor": anchor, "pos_pair": pos_pair}

    # def __getitem__(self, idx):
    #     pos1, pos2 = self.samples[idx]
    #     img1 = self._load_image(pos1)
    #     img2 = self._load_image(pos2)

    #     if self.transforms:
    #         img1 = self.transforms(img1)
    #         img2 = self.transforms(img2)
      

        # return {"image1": img1, "image2": img2}


#%%


def __getitem__(self, idx):
        position= self.samples[idx]
        anchor = self._load_image(position)


        pos_pair = self.transforms(anchor)

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
      

        return {"anchor": anchor, "pos_pair": img2}

#%% 
# Attempt #2 this block 
from pathlib import Path
import torch
from torch.utils.data import Dataset
from iohub.ngff import open_ome_zarr
import numpy as np
from monai.transforms import Compose, RandRotate, RandSpatialCrop, RandWeightedCrop

## in this case samples is all the possible positions
class PositivePairDataset(Dataset):
    def __init__(self, zarr_path, samples, patch_size, transforms=None):
        """
        Args:
            zarr_path (str or Path): path to your .zarr plate
            samples (list of tuples): list of positions
                each element: (row, col, field)
            patch_size (tuple): final YX patch size
            transforms (monai.transforms.Compose, optional): optional augmentation pipeline
        """
        self.zarr_path = Path(zarr_path)
        self.samples = samples
        self.patch_size = patch_size
        
        if transforms is None:
            self.transforms = Compose([
                RandSpatialCrop(roi_size = (patch_size[0]*1.41,patch_size[1]*1.41), max_roi_size=None, random_center=True, random_size=False, lazy=False),
                RandRotate(range_x=30, prob=0.5, keep_size=True, mode="bilinear"),
                RandSpatialCrop(roi_size = patch_size, max_roi_size=None, random_center=True, random_size=False, lazy=False),
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def _load_image(self, position):
        row, col, field = position
        with open_ome_zarr(self.zarr_path / str(row) / str(col) / str(field), mode="r") as store:
            arr = store["0"]  # shape (t=1, c=3, z=1, y, x)
            image = arr[0, :, 0, :, :]  # select t=0, all channels, z=0
            return torch.tensor(image, dtype=torch.float32)

    def __getitem__(self, idx):
        position = self.samples[idx]
        anchor = self._load_image(position)  # shape: (C, H, W)
        anchor = RandWeightedCrop(spatial_size= self.patch_size, num_samples=1, weight_map=2, lazy=False)
        # Apply transforms to create positive pair
        pos_pair = self.transforms(anchor)
        
        return {"anchor": anchor, "pos_pair": pos_pair}
#%%
#attempt number 3 using all positions
import torch
from torch.utils.data import Dataset
from pathlib import Path
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from monai.transforms import Compose, RandSpatialCrop, RandRotate, RandWeightedCrop
import numpy as np
import random

class PositivePairDataset(Dataset):
    def __init__(self, zarr_path, patch_size, split="train", val_fraction=0.1, transforms=None, seed=42):
        """
        Args:
            zarr_path (str or Path): path to your .zarr plate
            patch_size (tuple): final YX patch size
            split (str): "train" or "val"
            val_fraction (float): fraction of samples to reserve for validation
            transforms (monai.transforms.Compose, optional): augmentation pipeline
            seed (int): random seed for reproducibility
        """
        self.zarr_path = Path(zarr_path)
        self.patch_size = patch_size

        # --- get all positions from HCS plate ---
        all_positions = self._get_all_positions()
        random.Random(seed).shuffle(all_positions)

        # --- train/val split ---
        split_idx = int(len(all_positions) * (1 - val_fraction))
        if split == "train":
            self.samples = all_positions[:split_idx]
        else:
            self.samples = all_positions[split_idx:]

        # --- transforms ---
        if transforms is None:
            self.transforms = Compose([
                RandSpatialCrop(roi_size=(patch_size[0]*1.41, patch_size[1]*1.41), random_center=True, random_size=False),
                RandRotate(range_x=30, prob=0.5, keep_size=True, mode="bilinear"),
                RandSpatialCrop(roi_size=patch_size, random_center=True, random_size=False),
            ])
        else:
            self.transforms = transforms

    def _get_all_positions(self):
        """Traverse the OME-Zarr HCS plate and return all positions (row, col, field)."""
        store = parse_url(self.zarr_path, mode="r").store
        reader = Reader(store)
        nodes = list(reader())

        positions = []
        for row_key, row_node in nodes[0].children.items():
            for col_key, col_node in row_node.children.items():
                for field_key in col_node.children.keys():
                    positions.append((int(row_key), int(col_key), int(field_key)))
        return positions

    def __len__(self):
        return len(self.samples)

    def _load_image(self, position):
        row, col, field = position
        with open_ome_zarr(self.zarr_path / str(row) / str(col) / str(field), mode="r") as store:
            arr = store["0"]  # shape (t=1, c=3, z=1, y, x)
            image = arr[0, :, 0, :, :]  # select t=0, all channels, z=0
            return torch.tensor(image, dtype=torch.float32)

    def __getitem__(self, idx):
        position = self.samples[idx]
        anchor = self._load_image(position)  # shape: (C, H, W)

        # use 2nd channel (ER) as weight map for RandWeightedCrop
        weight_map = anchor[1].numpy()  
        cropper = RandWeightedCrop(spatial_size=self.patch_size, num_samples=1, weight_map=weight_map)
        anchor_cropped = cropper(anchor.unsqueeze(0))[0]  # add batch dim for monai API

        # Apply transforms to get positive pair
        pos_pair = self.transforms(anchor_cropped)

        return {"anchor": anchor_cropped, "pos_pair": pos_pair}






#%%
import pandas as pd
zarr_path = "/mnt/efs/aimbl_2025/student_data/S-RM/full_dataset/RM_project_ome.zarr"


# label_path = "/mnt/efs/aimbl_2025/student_data/S-RM/positive_pairs.csv"
# cols = ["row1", "col1", "fov1", "row2", "col2", "fov2"]

# df = pd.read_csv(label_path, usecols=cols)

# # ensure ints (avoids accidental float dtypes from CSVs)
# df[cols] = df[cols].astype(int)

# samples = [
#     ((r1, c1, f1), (r2, c2, f2))
#     for r1, c1, f1, r2, c2, f2 in df.itertuples(index=False, name=None)
# ]
# print(samples[:5])


# Example positive pairs (row, col, field)
# samples = [
#     ((1, 1, 1), (1, 1, 1)),  # first positive pair
#     ((1, 1, 1), (1, 1, 1)),  # second positive pair
#     ((1, 2, 1), (1, 2, 1)),  # third positive pair
# ]
samples = [
    (1, 1, 1),
    (1, 1, 1),
    (1, 2, 1),
]


# Instantiate dataset
dataset = PositivePairDataset(zarr_path,samples, patch_size=(384, 384))

print(f"Dataset length: {len(dataset)}")

# Test first item
item = dataset[0]
print("anchor shape:", item["anchor"].shape)
print("pos_pair shape:", item["pos_pair"].shape)

 #%%
## all below copied from demo  data loader
class HCSDataModule(LightningDataModule):
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
#%%
## working data module 
import pytorch_lightning as pl

class HCSDataModule(pl.LightningDataModule):
    def __init__(self, zarr_path, samples, patch_size=(256,256), batch_size=8, num_workers=4):
        super().__init__()
        self.zarr_path = zarr_path
        self.samples = samples
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers


        self.train_transforms = None ## using the transforms in the dataset
        self.val_transforms = Compose([]) ## setting transforms to none for val 

    def setup(self, stage=None):
        # Split samples into train/val
        split_idx = int(0.8 * len(self.samples))
        train_samples = self.samples[:split_idx]
        val_samples = self.samples[split_idx:]

        # Create Dataset objects
        self.train_dataset = PositivePairDataset(
            self.zarr_path, train_samples, patch_size=self.patch_size, transforms=self.train_transforms
        )
        self.val_dataset = PositivePairDataset(
            self.zarr_path, val_samples, patch_size=self.patch_size, transforms=self.val_transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
#%%