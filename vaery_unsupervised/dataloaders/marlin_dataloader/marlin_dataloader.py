#%%
%load_ext autoreload
%autoreload 2
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

T = 60
# H = 150
HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/')
KEY_FL = 'fluorescence'
KEY_SEG = 'segmentation'

from monai.transforms import (
    Flip,
    Rotate90,
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



# shift_in_both_chan
MEAN_OVER_DATASET = 13
STD_OVER_DATASET = 11
transform = Compose([
    Flip(spatial_axis=0),
    Rotate90(spatial_axes=(0, 2)),
])

transforms = Compose([
    NormalizeIntensity(
        subtrahend=MEAN_OVER_DATASET,
        divisor=STD_OVER_DATASET,
        nonzero=False,
        channel_wise=False,
        dtype = np.float32,
    ),
    Lambda(func=center_image,
           inv_func=None,
           track_meta=True)
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
 # TODO make this a transform

    def __getitem__(self, index: int):
        metadata_sample = self.metadata.iloc[index]
        # Get indices to find files
        gene_id, grna_id, grna_file_trench_index, timepoint = metadata_sample[['gene_id', 'oDEPool7_id', 'grna_file_trench_index', 'timepoints']]

        # Using only the next timepoint
        filename = self.data_path / f"{gene_id}/{grna_id}.hdf5"
        with h5py.File(filename, 'r') as f:
            img_fl_anchor = f[KEY_FL][grna_file_trench_index, timepoint]
            img_fl_pos = f[KEY_FL][grna_file_trench_index, (timepoint+1)%T] # NO CORRECTION
            img_seg_anchor = f[KEY_SEG][grna_file_trench_index, timepoint]
            img_seg_pos = f[KEY_SEG][grna_file_trench_index, (timepoint+1)%T] # NO CORRECTION

        # Set your desired output size
        # TODO DECIDE IF INCLUDING SEG
        img_fl_anchor = self.center_image(img_fl_anchor)
        img_fl_pos = self.center_image(img_fl_pos)
        img_seg_anchor = self.center_image(img_seg_anchor)
        img_seg_pos = self.center_image(img_seg_pos)

        # TODO: Apply transformations!
        if self.transform:
            NotImplementedError("Transformations are not implemented")

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
        COLS_TO_LOAD = ['gene_id', 'oDEPool7_id', 'grna_file_trench_index', 'timepoints']
        self.metadata = (pd
            .read_pickle(self.metadata_path)
            [COLS_TO_LOAD]
        )

    def _prepare_data(self):
        # Logica para abrir la metadta dataframe, filter y shuffle and split
        # indices_all = self.metadata.index.tolist()
        indices_shuffled = np.random.permutation(self.metadata.index.tolist())
        indices_train = indices_shuffled[:int(len(indices_shuffled) * self.split_ratio)]
        indices_val = indices_shuffled[int(len(indices_shuffled) * self.split_ratio):]
        self.metadata_train = self.metadata.loc[indices_train]
        self.metadata_val = self.metadata.loc[indices_val]

    def setup(self, stage):
        if stage == 'fit': # includes validation
            self.dataset = MarlinDataset(
                data_path=self.data_path,
                metadata=self.metadata_train,
                transform=...
            )
        elif stage == 'validate':
            self.dataset = MarlinDataset(
                data_path=self.data_path,
                metadata=self.metadata_val,
                transforms=...
            )
        else: # 'test' or 'predict'
            self.dataset = MarlinDataset(
                data_path=self.data_path,
                metadata=self.metadata,
                transform=...
            )

    def train_dataloader(self): 
        return DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )

    # Whenever called, I need to change mode
    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
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

#%%
HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/')
data_module = MarlinDataModule(
    data_path=HEADPATH/'Ecoli_lDE20_Exps-0-1/',
    metadata_path=HEADPATH/'2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl',
    split_ratio=0.8,
    batch_size=1024,
    num_workers=4,
    prefetch_factor=2,
    transforms=[...]
)

#%%
data_module._prepare_data()
#%%
data_module.setup(stage='fit')

# Transforms
 # Reflection over either axis
 # Shuffling of cells over the trench?
 # 
# %%
img_anchor, img_pos =data_module.dataset[45000].values()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img_anchor, cmap='gray')
ax[0].set_title('Anchor Image')
ax[0].axis('off')
ax[1].imshow(img_pos, cmap='gray')
ax[1].set_title('Positive Image')
ax[1].axis('off')
plt.show()
#%%

# transforms(img_anchor)
center_image(img_anchor)
#%%
metadata_sample = data_module.dataset.metadata.iloc[3001]
gene_id, grna_id, grna_file_trench_index, timepoint = metadata_sample[['gene_id', 'oDEPool7_id', 'grna_file_trench_index', 'timepoints']]
filename = data_module.dataset.data_path / f"{gene_id}/{grna_id}.hdf5"
with h5py.File(filename, 'r') as f:
    imgs = f[KEY_FL][grna_file_trench_index]
    print(imgs.shape)

#%%
metadata = pd.read_pickle(HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl')
metadata