#%%
import lightning as L
from pathlib import Path
from typing import Callable, Literal, Sequence, Optional
import pandas as pd
from torch.utils.data import Dataset
import h5py
import numpy as np

T = 60
H = 150
HEADPATH = '/mnt/efs/aimbl_2025/student_data/S-GL/'
KEY_FL = 'fluorescence'
KEY_SEG = 'segmentation'

class MarlinDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 metadata_path: str,
                #  train_image_width: int,
                #  sampling_type: str = None,
                 transform: Optional[Callable] = None):
        self.data_path = data_path
        self.metadata_path = metadata_path
        # self.train_image_width = train_image_width
        # self.sampling_type = sampling_type
        self.metadata = pd.read_pickle(metadata_path)
        self.transform = transform

    def __len__(self):
        return len(self.metadata) # verify

    def center_image(self, img, D):
            h, w = img.shape
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
        img_fl_anchor = self.center_image(img_fl_anchor, H)
        img_fl_pos = self.center_image(img_fl_pos, H)
        img_seg_anchor = self.center_image(img_seg_anchor, H)
        img_seg_pos = self.center_image(img_seg_pos, H)

        return {'anchor': img_fl_anchor, 'positive': img_fl_pos}
#%%
dataset = MarlinDataset(
    data_path=Path(HEADPATH+'Ecoli_lDE20_Exps-0-1/'),
    metadata_path=Path(HEADPATH+'2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl'),
    transform=None
)
#%%
dataset
#%%
metadata = pd.read_pickle(HEADPATH+'2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering.pkl')
metadata
COLS_TO_KEEP = ['Multi-Experiment Phenotype Trenchid', 'oDEPool7_id', 
                'Gene', 'gene_id', 'Experiment #', 'grna_file_trench_index']
#%%
%load_ext autoreload
%autoreload 2
#%%
import marlin_dataloader.utils as mdu
metadata_expanded = mdu.expand_metadata_on_time(metadata, n_timepoints=T, cols_to_keep=COLS_TO_KEEP)
#%%
# metadata_expanded.to_pickle(HEADPATH+'2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl')
metadata_expanded = pd.read_pickle(HEADPATH+'2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl')
metadata_expanded
#%%
dataset.metadata.loc[:, COLS_TO_KEEP].reset_index()

expanded_metadata = pd.DataFrame(
    dataset.metadata.loc[:, COLS_TO_KEEP].reset_index()
    .apply(lambda x: x.repeat(T).reset_index(drop=True))
)
expanded_metadata['timepoint'] = expanded_metadata.groupby(expanded_metadata.index // T).cumcount()
# expanded_metadata.head()
#%%
expanded_metadata.to_pickle(HEADPATH+'2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl')
#%%
# Get total size in MB of dataset.metadata
metadata_size = expanded_metadata.memory_usage(deep=True).sum() / (1024 ** 2)
print(f"Total size of dataset.metadata: {metadata_size:.2f} MB")

# dataset.metadata
#%%
class DemoDataModule(L.LightningDataModule):
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
        self.metadata = pd.read_pickle(metadata_path)
        self.indices = tuple(
            (str(gene_id), str(oDEPool7_id), str(grna_file_trench_index))
            for gene_id, oDEPool7_id, grna_file_trench_index in
            self.metadata.loc[:, ['gene_id', 'oDEPool7_id', 'grna_file_trench_index']].itertuples(index=False, name=None)
        )
        self.split_ratio = split_ratio
    

        print(self.indices)
        print(len(self.indices))
            
        def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        dataset_settings = self._base_dataset_settings
        if stage in ("fit", "validate"):
            self._setup_fit(dataset_settings)
        elif stage == "test":
            self._setup_test(dataset_settings)
        elif stage == "predict":
            self._setup_predict(dataset_settings)
        else:
            raise NotImplementedError(f"{stage} stage")
        # Let's just use last timepoints to start with
        # Random shuffle 

        # self.train_datapaths = ...
        # self.val_datapath = self.val_dataset
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