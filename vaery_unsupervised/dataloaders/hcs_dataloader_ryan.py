#%%
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
from monai.transforms import Compose, RandSpatialCrop, RandRotate, RandWeightedCrop, CenterSpatialCrop, NormalizeIntensity


class ContrastiveHCSDataset(Dataset):
    def __init__(self, positions: list[Position], source_channel_names: list[str],weight_channel_name: str = 'nuclei', crop_size: tuple[int, int]=(256, 256), 
                 crops_per_position: int=4, normalization_transform: Compose| list[Callable]=[], anchor_augmentations: Callable| Compose=[], positive_augmentations: Compose| Callable=[]):
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
        self.weight_channel_name = weight_channel_name
        self.crop_size = crop_size
        self.crops_per_position = crops_per_position
        self.normalization_transform = Compose(normalization_transform)
        self.anchor_augmentations = Compose(anchor_augmentations)
        self.positive_augmentations = Compose(positive_augmentations)
        
        # self.positive_augmentations = Compose([
            #     RandSpatialCrop(roi_size = (crop_size[0]*1.41,crop_size[1]*1.41), max_roi_size=None, random_center=True, random_size=False, lazy=False),
            #     RandRotate(range_x=30, prob=0.5, keep_size=True, mode="bilinear"),
            #     RandSpatialCrop(roi_size = crop_size, max_roi_size=None, random_center=True, random_size=False, lazy=False),
            # ])
        # make mapping between the crop_per_position and the actual index in the dataset
        # repeat the number of positions by the number of crops per position
        self.all_positions = [pos for pos in positions for _ in range(self.crops_per_position)]

    def __getitem__(self, index):

        #TODO open the 
        # List of iohub Position objects
        position = self.all_positions[index]
        channel_names = position.channel_names
        source_channel_indices = [position.channel_names.index(ch) for ch in self.source_channel_names]
        # mapping of the weighted_channel_index to the source_channel_indices

        #TODO open the position
        img_cyx = position[0].oindex[0, source_channel_indices, 0]
        weight_channel_index = self.source_channel_names.index('nuclei')
        weight_img = img_cyx[weight_channel_index:weight_channel_index+1]  # Shape: (1, Y, X)
        
        random_weighted_crop = RandWeightedCrop(
            spatial_size=(self.crop_size[0]*1.41, self.crop_size[1]*1.41),  # (H, W) for 2D spatial crop
            weight_map=weight_img,  # Remove channel dim, now (Y, X)
            num_samples=1
        )
        crop_img = random_weighted_crop(img_cyx)[0]
        
        #TODO apply the transform and the normalizations
        #compose the the monai transforms
        #image -min (over max-min)

        # norm_img = NormalizeIntensity()
        
        max_per = torch.quantile(crop_img, 0.95)
        min_per = torch.quantile(crop_img, 0.05)

        norm_img = (crop_img - min_per) / (max_per - min_per)

        if torch.isnan(norm_img).any():
            norm_img = torch.nan_to_num(norm_img, nan=0.0, posinf=1.0, neginf=0.0)

        anchor = self.anchor_augmentations(norm_img)
        positive = self.positive_augmentations(anchor)

        # anchor = anchor[:,None,...]
        # positive = positive[:,None,...]


        #TODO return the anchor and positive
        return {
            # "anchor": anchor[0][:,None,...],
            # "positive": positive[0][:,None,...],
            # "anchor": anchor[0],
            # "positive": positive[0],
            "anchor": anchor,
            "positive": positive,
            # "fov_id": position.name #TODO check if we need the fov_id
        }
    
    def __len__(self):
        return len(self.all_positions)


class HCSDataModule(pl.LightningDataModule):
    def __init__(
            self, ome_zarr_path, source_channel_names, weight_channel_name, crop_size=(256, 256),
            crops_per_position=4, batch_size=32, num_workers=4, 
            split_ratio=0.8, random_state=42, 
            normalization_transform=[], augmentations=[]
        ):
        super().__init__()
        self.ome_zarr_path = ome_zarr_path
        self.source_channel_names = source_channel_names
        self.crop_size = crop_size
        self.crops_per_position = crops_per_position
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.weight_channel_name = weight_channel_name
        # self.train_ratio = train_ratio
        # self.val_ratio = val_ratio
        # self.test_ratio = test_ratio
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
            positions= positions[:num_train_fovs],
            crops_per_position=self.crops_per_position,
            source_channel_names= ['mito','er','nuclei'],
            normalization_transform = self.normalization_transform,
            anchor_augmentations= [],
            # positive_augmentations= [],
            weight_channel_name= self.weight_channel_name,
            positive_augmentations= Compose([
                RandRotate(range_x=30, prob=1.0, keep_size=True, mode="bilinear", padding_mode="zeros"),
                CenterSpatialCrop(roi_size=self.crop_size),
            ]),
        )
        self.val_dataset = ContrastiveHCSDataset(
            positions = positions[num_train_fovs:],
            crops_per_position=self.crops_per_position,
            source_channel_names= ['mito','er','nuclei'],
            normalization_transform = [],
            anchor_augmentations= [],
            positive_augmentations= Compose([
                RandRotate(range_x=30, prob=1.0, keep_size=True, mode="bilinear", padding_mode="zeros"),
                CenterSpatialCrop(roi_size=self.crop_size)
                ]),
            weight_channel_name= self.weight_channel_name,
            
        )
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         shuffle=False, num_workers=self.num_workers)
    #%%