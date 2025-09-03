#%%
import lightning as L
from pathlib import Path
from typing import Callable, Literal, Sequence, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

T = 60 # Number of timepoints
KEY_FL = 'fluorescence'
KEY_SEG = 'segmentation'

from monai.transforms import (
    Compose,
    NormalizeIntensity,
    Lambda,
)

def center_image(img):
    h, w = img.shape
    D = np.max([h, w])# TODO: Pass constants?
    out = np.zeros((D, D), dtype=img.dtype)
    y_offset = (D - h) // 2
    x_offset = (D - w) // 2
    out[y_offset:y_offset+h, x_offset:x_offset+w] = img
    return out

# Shift parameteres
MEAN_OVER_DATASET = 13
STD_OVER_DATASET = 11

transforms = Compose([
    NormalizeIntensity(
        subtrahend=MEAN_OVER_DATASET,
        divisor=STD_OVER_DATASET,
        nonzero=False,
        channel_wise=False,
        dtype = np.float32,
    ),
    # Lambda(func=center_image,
    #        inv_func=None,
    #        track_meta=True)
])

class MarlinDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 metadata: pd.DataFrame,
                 transform: Optional[Callable] = None):
        super().__init__()
        self.data_path = data_path
        self.metadata = metadata
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
 
    def __getitem__(self, index: int):
        metadata_sample = self.metadata.iloc[index]
        # Get indices to find files
        gene_id, grna_id, grna_file_trench_index, timepoint = metadata_sample[['gene_id', 'oDEPool7_id', 'grna_file_trench_index', 'timepoints']]
        # Using only the next timepoint, reset to the first timepoint if chose the last timepoint
        filename = self.data_path / f"{gene_id}/{grna_id}.hdf5"

    
        with h5py.File(filename, 'r') as f:
            #### Two timepoints
            img_fl_anchor = f[KEY_FL][grna_file_trench_index, timepoint]
            img_fl_pos = f[KEY_FL][grna_file_trench_index, (timepoint+1)%T] # NO CORRECTION
            # img_seg_anchor = f[KEY_SEG][grna_file_trench_index, timepoint]
            # img_seg_pos = f[KEY_SEG][grna_file_trench_index, (timepoint+1)%T] # NO CORRECTION


        # standardize the pixel values and convert to float 32
        img_fl_anchor = img_fl_anchor.astype(np.float32)
        img_fl_pos = img_fl_pos.astype(np.float32)

        perc_2_anchor = np.percentile(img_fl_anchor, 2)
        perc_98_anchor = np.percentile(img_fl_anchor, 98)
        img_fl_anchor = (img_fl_anchor - perc_2_anchor) / (perc_98_anchor - perc_2_anchor)
        img_fl_anchor = 2 * img_fl_anchor - 1

        perc_2_pos = np.percentile(img_fl_pos, 2)
        perc_98_pos = np.percentile(img_fl_pos, 98)
        img_fl_pos = (img_fl_pos - perc_2_pos) / (perc_98_pos - perc_2_pos)
        img_fl_pos = 2 * img_fl_pos - 1
        # img_fl_anchor = (img_fl_anchor - np.mean(img_fl_anchor)) / np.std(img_fl_anchor)
        # img_fl_pos = (img_fl_pos - np.mean(img_fl_pos)) / np.std(img_fl_pos)
        # Set your desired output size
        # TODO DECIDE IF INCLUDING SEG
        # TODO make this a transform
        img_fl_anchor = center_image(img_fl_anchor)
        img_fl_pos = center_image(img_fl_pos)
        # img_seg_anchor = self.center_image(img_seg_anchor)
        # img_seg_pos = self.center_image(img_seg_pos)

        img_fl_anchor = img_fl_anchor[None,...]
        img_fl_pos = img_fl_pos[None,...]
        
        # if self.transform: # TODO ADD CENTERING HERE!
        #     img_fl_anchor = self.transform(img_fl_anchor)
        #     img_fl_pos = self.transform(img_fl_pos)

        # TODO: CASE FOR TESTING/PREDICTING
        # if self.train_mode:
        return {'anchor': img_fl_anchor, 'positive': img_fl_pos}
        # else:
            # return {'anchor': img_fl_anchor}
        
class MarlinDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        metadata_path: str,
        split_ratio:int,
        batch_size:int,
        num_workers:int,
        prefetch_factor:int=2,
        transforms: list = None
    ):
        super().__init__()
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.transforms = transforms

        # Open the metadata
        COLS_TO_LOAD = ['gene_id', 'oDEPool7_id', 'grna_file_trench_index', 'timepoints']
        self.metadata = (
            pd.read_pickle(self.metadata_path)
            [COLS_TO_LOAD]
        )

    def _prepare_data(self):
        # Shuffle and split the metadata
        indices_shuffled = np.random.permutation(self.metadata.index.tolist())
        indices_train = indices_shuffled[:int(len(indices_shuffled) * self.split_ratio)]
        indices_val = indices_shuffled[int(len(indices_shuffled) * self.split_ratio):]
        self.metadata_train = self.metadata.loc[indices_train]
        self.metadata_val = self.metadata.loc[indices_val]

    def setup(self, stage):
        if stage == 'fit': # includes validation
            self.train_dataset = MarlinDataset(
                data_path=self.data_path,
                metadata=self.metadata_train,
                transform=self.transforms
            )
            self.val_dataset = MarlinDataset(
                data_path=self.data_path,
                metadata=self.metadata_val,
                transform=self.transforms
            )
        elif stage == 'validate':
            self.dataset = MarlinDataset(
                data_path=self.data_path,
                metadata=self.metadata_val,
                transforms=self.transforms
            )
        else: # 'test' or 'predict'
            self.dataset = MarlinDataset(
                data_path=self.data_path,
                metadata=self.metadata,
                transform=self.transforms
            )

    def train_dataloader(self): 
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )

    # Whenever called, I need to change mode
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )