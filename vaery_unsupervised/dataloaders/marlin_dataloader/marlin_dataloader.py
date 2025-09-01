#%%
%load_ext autoreload
%autoreload 2
import lightning as L
from pathlib import Path
from typing import Callable, Literal, Sequence, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import h5py
import numpy as np

T = 60
# H = 150
HEADPATH = '/mnt/efs/aimbl_2025/student_data/S-GL/'
KEY_FL = 'fluorescence'
KEY_SEG = 'segmentation'

class MarlinDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 metadata_path: str,
                 train_mode:bool=True,
                 split_ratio: float = 0.8,
                #  train_image_width: int,
                #  sampling_type: str = None,
                 transform: Optional[Callable] = None):
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.train_mode = train_mode
        self.split_ratio = split_ratio
        # self.train_image_width = train_image_width
        # self.sampling_type = sampling_type
        self.metadata = pd.read_pickle(metadata_path)
        self.transform = transform

        indices_all = self.metadata.index.tolist()
        np.random.shuffle(indices_all) # TODO
        self.train_indices = indices_all[:int(len(indices_all) * self.split_ratio)]
        self.val_indices = indices_all[int(len(indices_all) * self.split_ratio):]
        if self.train_mode:
            self.metadata = self.metadata.iloc[train_indices]
        else:
            self.metadata = self.metadata.iloc[val_indices]

    def __len__(self):
        return len(self.metadata)

    def center_image(self, img):
            h, w = img.shape
            D = np.max([h, w])
            out = np.zeros((D, D), dtype=img.dtype)
            y_offset = (D - h) // 2
            x_offset = (D - w) // 2
            out[y_offset:y_offset+h, x_offset:x_offset+w] = img
            return out

    def __getitem__(self, index: int):
        
        metadata_sample = self.metadata.iloc[index]
        print(metadata_sample[['gene_id', 'oDEPool7_id', 'grna_file_trench_index']])
        gene_id, grna_id, grna_file_trench_index, timepoint = metadata_sample[['gene_id', 'oDEPool7_id', 'grna_file_trench_index', 'timepoints']]

        # Using only the last timepoint in the meantime
        filename = self.data_path / f"{gene_id}/{grna_id}.hdf5"
        with h5py.File(filename, 'r') as f:
            img_fl_anchor = f[KEY_FL][grna_file_trench_index, timepoint]
            img_fl_pos = f[KEY_FL][grna_file_trench_index, (timepoint+1)%T]
            img_seg_anchor = f[KEY_SEG][grna_file_trench_index, timepoint]
            img_seg_pos = f[KEY_SEG][grna_file_trench_index, (timepoint+1)%T]

          # Set your desired output size
        img_fl_anchor = self.center_image(img_fl_anchor)
        img_fl_pos = self.center_image(img_fl_pos)
        img_seg_anchor = self.center_image(img_seg_anchor)
        img_seg_pos = self.center_image(img_seg_pos)

        return {'anchor': img_fl_anchor, 'positive': img_fl_pos}
#%%
dataset = MarlinDataset(
    data_path=Path(HEADPATH+'Ecoli_lDE20_Exps-0-1/'),
    metadata_path=Path(HEADPATH+'2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl'),
    transform=None
)
#%%
img_anc, img_pos = dataset[3001].values()

import matplotlib.pyplot as plt
# Visualize the images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_anc, cmap='gray')
plt.title('Anchor Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_pos, cmap='gray')
plt.title('Positive Image')
plt.axis('off')
plt.show()

#%%
class MarlinDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        metadata_path: str,
        batch_size:int,
        split_ratio:int,
        num_workers:int,
        prefetch_factor:int=2,
        transforms: list = None#Optional[Callable] = None
    ):
        super().__init__()
        self.data_path = data_path
        # parse indices
        self.metadata_path = metadata_path
        self.metadata = pd.read_pickle(metadata_path)
        self.batch_size = batch_size
        self.indices = self.metadata.index.tolist()
        self.split_ratio = split_ratio
    
        print(self.indices)
        print(len(self.indices))

        self.train_indices, self.val_indices = train_test_split(self.indices, test_size=self.split_ratio)

    def setup(self, stage: str):
        if stage == "fit" or stage == "train":
            self.dataset = MarlinDataset(self.data_paths, self.train_indices)
        elif stage == "val" or stage == "validate":
            self.dataset = MarlinDataset(self.data_paths, self.val_indices)
        elif stage == "predict":
        NotImplementedError("This method is not implemented")

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor)

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor)

#%%

HEADPATH = '/mnt/efs/aimbl_2025/student_data/S-GL/'
data_module = DemoDataModule(
    data_path=Path(HEADPATH+'Ecoli_lDE20_Exps-0-1/'),
    metadata_path = Path(HEADPATH+'2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering.pkl'),
    batch_size=32,
    num_workers=4,
    prefetch_factor=2,
    transforms=[...]
)

#%%
data_module.indices

#%%
    def prepare_data(self):
        return None
    def setup(self, stage:str):
        if stage == "fit":
            print(0)
            # self.train_dataset = MyDataset(train=True)
        elif stage == "validate":
            print(1)
            # self.val_dataset = MyDataset(train=False)
        

    def train_dataloader(self):

def __get_item(self, index):
    sample = open_ome_zarr()

    if self.transform:
        NotImplementedError("")
        # self.transform a list of transforms [crop, rotation, flip, normalize]
        # List of transforms
        sample - self.transform(sample)

        # return anchor, and positive, and positive augmentation.


    return sample

# Transforms
 # Reflection over either axis
 # Shuffling of cells over the trench?
 # 