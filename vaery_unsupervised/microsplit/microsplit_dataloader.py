from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
from iohub import open_ome_zarr
from iohub.ngff import Position
from monai.data import set_track_meta
from monai.transforms import Compose
from torch.utils.data import DataLoader, Dataset
from monai.transforms import Compose, RandSpatialCrop

from utils import extract_overlapping_patches_grid


class MicroSplitHCSTrainingDataset(Dataset):
    def __init__(
        self,
        positions: list[Position],
        source_channel_names: list[str],
        crop_size: tuple[int, int] = (128, 128),
        crops_per_position: int = 4,
        transforms: list[Callable] = None,
        mix_coeff_range: tuple[float, float] | None = None,
    ) -> None:
        """
        Parameters:
            positions: List of iohub.ngff.Position objects
            source_channel_names: Channel names to use
            crop_size: Size of random crops
            crops_per_position: Number of crops to generate per position per epoch
            
        """
        self.positions = positions
        self.source_channel_names = source_channel_names
        self.crop_size = crop_size
        self.crops_per_position = crops_per_position
        self.transforms = Compose(transforms) if transforms else None
        self.mix_coeff_range = mix_coeff_range

        # get all positions from file
        self.all_positions = [pos for pos in positions for _ in range(self.crops_per_position)]

    def __getitem__(self, index):
        # get multichannel image
        position = self.all_positions[index]
        source_channel_indices = [position.channel_names.index(ch) for ch in self.source_channel_names]
        multi_ch_img = position[0].oindex[0, source_channel_indices, 0]

        # quantile normalization on the full image
        max_per = np.quantile(multi_ch_img, 0.99, axis=(1, 2), keepdims=True)
        min_per = np.quantile(multi_ch_img, 0.01, axis=(1, 2), keepdims=True)
        multi_ch_img = (multi_ch_img - min_per) / (max_per - min_per)

        # filter out empty patches setting them to zeros
        if np.isnan(multi_ch_img).any():
            multi_ch_img = np.nan_to_num(multi_ch_img, nan=0.0, posinf=1.0, neginf=0.0)

        # convert to tensor
        multi_ch_img = torch.from_numpy(multi_ch_img).float()
        
        # apply transforms if any
        if self.transforms:
            multi_ch_img = self.transforms(multi_ch_img)

        # crop image
        cropper = RandSpatialCrop(roi_size=self.crop_size)
        multi_ch_img = cropper(multi_ch_img)
        
        # get superimposed image
        if self.mix_coeff_range is not None:
            mix_coeffs = torch.empty(multi_ch_img.shape[0]).uniform_(*self.mix_coeff_range)
        else:
            mix_coeffs = torch.ones(multi_ch_img.shape[0])
        mixed_img = torch.mean(multi_ch_img * mix_coeffs[:, None, None], dim=0, keepdim=True)

        return mixed_img, multi_ch_img
    
    def __len__(self):
        return len(self.all_positions)


class MicroSplitHCSInferenceDataset(Dataset):
    def __init__(
        self,
        positions: list[Position],
        source_channel_names: list[str],
        crop_size: tuple[int, int] = (128, 128),
        crops_per_position: int = 4,
    ) -> None:
        """
        Parameters:
            positions: List of iohub.ngff.Position objects
            source_channel_names: Channel names to use
            crop_size: Size of random crops
            crops_per_position: Number of crops to generate per position per epoch
            
        """
        self.positions = positions
        self.source_channel_names = source_channel_names
        self.crop_size = crop_size
        self.crops_per_position = crops_per_position

        # get all positions from file
        self.all_positions = [pos for pos in positions for _ in range(self.crops_per_position)]

    def __getitem__(self, index):
        # get multichannel image
        position = self.all_positions[index]
        source_channel_indices = [position.channel_names.index(ch) for ch in self.source_channel_names]
        multi_ch_img = position[0].oindex[0, source_channel_indices, 0]

        # quantile normalization on the full image
        max_per = np.quantile(multi_ch_img, 0.99, axis=(1, 2), keepdims=True)
        min_per = np.quantile(multi_ch_img, 0.01, axis=(1, 2), keepdims=True)
        multi_ch_img = (multi_ch_img - min_per) / (max_per - min_per)

        # filter out empty patches setting them to zeros
        if np.isnan(multi_ch_img).any():
            multi_ch_img = np.nan_to_num(multi_ch_img, nan=0.0, posinf=1.0, neginf=0.0)

        # convert to tensor
        multi_ch_img = torch.from_numpy(multi_ch_img).float()

        # crop image in a grid of overlapping patches
        multi_ch_patches, coords = extract_overlapping_patches_grid(
            img=multi_ch_img.numpy(),
            patch_size=self.crop_size,
            overlap=(self.crop_size[0] // 2, self.crop_size[1] // 2)
        )
        patches_info = {"coords": coords, "idx": index}

        # get superimposed image
        mixed_img = torch.mean(multi_ch_patches, dim=1, keepdim=True)

        return mixed_img, multi_ch_img, patches_info
    
    def __len__(self):
        return len(self.all_positions)


class MicroSplitHCSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ome_zarr_path: str,
        source_channel_names: list[str],
        crop_size: int = (128, 128),
        crops_per_position: int = 4,
        random_cropping: bool = True,
        batch_size: int = 32,
        num_workers: int = 3,
        split_ratios: list[float, float, float] = [0.9, 0.1, 0.1], 
        random_state: int = 42,
        augmentations: list[Callable] = [],
        mix_coeff_range: tuple[float, float] = (0.0, 1.0)
    ) -> None:
        super().__init__()
        self.ome_zarr_path = ome_zarr_path
        self.source_channel_names = source_channel_names
        self.crop_size = crop_size
        self.crops_per_position = crops_per_position
        self.random_cropping = random_cropping
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratios = split_ratios
        self.random_state = random_state
        self.augmentations = augmentations
        self.mix_coeff_range = mix_coeff_range

    def _set_fit_global_state(self, num_positions: int) -> torch.Tensor:
        # disable metadata tracking in MONAI for performance
        set_track_meta(False)
        # shuffle positions, randomness is handled globally
        return torch.randperm(num_positions)

    def setup(self, stage: str = None):
        # split FOVs for train and val
        plate = open_ome_zarr(self.ome_zarr_path, mode="r")
        positions = [pos for _, pos in plate.positions()]
        shuffled_indices = self._set_fit_global_state(len(positions))
        positions = list(positions[i] for i in shuffled_indices)
        num_train_fovs = int(len(positions) * self.split_ratios[0])
        num_val_fovs = int(len(positions) * self.split_ratios[1])
        num_test_fovs = len(positions) - num_train_fovs - num_val_fovs

        # initialize datasets
        self.train_dataset = MicroSplitHCSTrainingDataset(
            positions= positions[:num_train_fovs],
            crop_size=self.crop_size,
            crops_per_position=self.crops_per_position,
            source_channel_names= ['mito', 'er', 'nuclei'],
            transforms=self.augmentations,
            mix_coeff_range=self.mix_coeff_range
        )
        self.val_dataset = MicroSplitHCSTrainingDataset(
            positions=positions[num_train_fovs:(num_train_fovs + num_val_fovs)],
            crop_size=self.crop_size,
            crops_per_position=self.crops_per_position,
            source_channel_names=['mito', 'er', 'nuclei'],
            transforms=None,
            mix_coeff_range=None
        )
        self.test_dataset = MicroSplitHCSInferenceDataset(
            positions=positions[(num_train_fovs + num_val_fovs):],
            crop_size=self.crop_size,
            crops_per_position=self.crops_per_position,
            source_channel_names=['mito', 'er', 'nuclei'],
        )
        
    def test_data_collate_fn(self, batch):
        # Flatten patches across the batch
        input_patches = torch.cat([item[0] for item in batch], dim=0)  # [B*N, C, H, W]
        target_patches = torch.cat([item[1] for item in batch], dim=0)  # [B*N, C, H, W]

        # Flatten coords in the same order
        p_info = []
        for item in batch:
            p_info.extend(item[2])
        
        return input_patches, target_patches, p_info

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
            num_workers=self.num_workers,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_data_collate_fn
        )
