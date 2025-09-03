from typing import Callable
import pytorch_lightning as pl
import torch
from iohub import open_ome_zarr
from iohub.ngff import Position
from monai.data import set_track_meta
from monai.transforms import Compose
from torch.utils.data import DataLoader, Dataset
from monai.transforms import Compose, RandRotate, RandSpatialCrop


class MicroSplitHCSDataset(Dataset):
    def __init__(
        self,
        positions: list[Position],
        source_channel_names: list[str],
        crop_size: tuple[int, int] = (128, 128),
        crops_per_position: int = 4,
        transforms: list[Callable] = None,
        mix_coeff_range: tuple[float, float] = (0.0, 1.0),
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
        max_per = torch.quantile(multi_ch_img, 0.99)
        min_per = torch.quantile(multi_ch_img, 0.01)
        multi_ch_img = (multi_ch_img - min_per) / (max_per - min_per)

        # filter out empty patches setting them to zeros
        if torch.isnan(multi_ch_img).any():
            multi_ch_img = torch.nan_to_num(multi_ch_img, nan=0.0, posinf=1.0, neginf=0.0)

        # crop image
        cropper = RandSpatialCrop(roi_size=self.crop_size)
        multi_ch_img = cropper(multi_ch_img)

        # get superimposed image
        mix_coeffs = torch.empty(multi_ch_img.shape[0]).uniform_(*self.mix_coeff_range)
        mixed_img = torch.mean(multi_ch_img * mix_coeffs[:, None], dim=0, keepdim=True)

        return {
            "input": mixed_img,
            "target": multi_ch_img,
        }
    
    def __len__(self):
        return len(self.all_positions)


class MicroSplitHCSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ome_zarr_path: str,
        source_channel_names: list[str],
        crop_size: int = (128, 128),
        crops_per_position: int = 4,
        batch_size: int = 32,
        num_workers: int = 3,
        split_ratio: float = 0.8,
        random_state: int = 42,
        augmentations: list[Callable] = []
    ) -> None:
        super().__init__()
        self.ome_zarr_path = ome_zarr_path
        self.source_channel_names = source_channel_names
        self.crop_size = crop_size
        self.crops_per_position = crops_per_position
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.augmentations = augmentations

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
        num_train_fovs = int(len(positions) * self.split_ratio)

        # initialize datasets
        self.train_dataset = MicroSplitHCSDataset(
            positions= positions[:num_train_fovs],
            crops_per_position=self.crops_per_position,
            source_channel_names= ['mito', 'er', 'nuclei'],
            transforms=self.augmentations,
        )
        self.val_dataset = MicroSplitHCSDataset(
            positions=positions[num_train_fovs:],
            crops_per_position=self.crops_per_position,
            source_channel_names=['mito', 'er', 'nuclei'],
            transforms=None
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
