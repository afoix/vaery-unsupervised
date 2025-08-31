#%%
from pathlib import Path
from iohub import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
import h5py
import pandas as pd
import numpy as np

# HEADPATH = Path('/home/lag36/scratch/lag36/Ecoli/2023-01-18_lDE20_Merged_Analysis/')
HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/')
metadata_path = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering.pkl'
output_zarr_store = HEADPATH / 'Ecoli_lDE20_Exps-0-1.zarr'
path_to_dataset = HEADPATH / 'Ecoli_lDE20_Exps-0-1'

def load_array_from_hdf5(
    file_idx: int,
    headpath: Path,
    prefix: str,
    key: str) -> np.ndarray:
        filepath = headpath / f"{prefix}{file_idx}.hdf5"
        with h5py.File(filepath, 'r') as f:
            images = f[key][:]
        return images

#%% 
metadata = pd.read_pickle(metadata_path)

position_keys = tuple(
    (str(gene_id), str(oDEPool7_id), str(grna_file_trench_index))
    for gene_id, oDEPool7_id, grna_file_trench_index in metadata.loc[:, ['gene_id', 'oDEPool7_id', 'grna_file_trench_index']].itertuples(index=False, name=None)
)
position_keys
#%%
gene_id, grna_id, grna_trench_id = position_keys[0]
sample_image = load_array_from_hdf5(
    file_idx=f'{grna_id}',
    headpath=path_to_dataset,
    prefix=f'{gene_id}/',
    key='fluorescence'
)
T, Y, X = sample_image.shape[1:]
C = 1
Z = 1
zarr_array_shape = (T, C, Z, Y, X)
print(zarr_array_shape)

#%%
PIXEL_SIZE_MICRONS = 0.105951961895293*2
create_empty_plate(store_path=output_zarr_store,
    position_keys = position_keys,
    channel_names = ['mkate'], #TODO: replace for the channel names
    shape = zarr_array_shape, #TODO: replace for the shape (T,C,Z,Y,X)
    chunks = zarr_array_shape, #TODO: this is important for training (T,C,Z,Y,X)
    scale = (1, 1, 1, PIXEL_SIZE_MICRONS, PIXEL_SIZE_MICRONS), #TODO: [Optional] replace for the scale in um. (T,C,Z,Y,X)
    dtype= np.float32,#np.float32, 
)
#%%
for gene_id, grna_id, grna_trench_id in position_keys:
    fl_image = load_array_from_hdf5(
    file_idx=f'{grna_id}',
    headpath=path_to_dataset,
    prefix=f'{gene_id}/',
    key='fluorescence'
    )

    seg_image = load_array_from_hdf5(
        file_idx=f'{grna_id}',
        headpath=path_to_dataset,
        prefix=f'{gene_id}/',
        key='segmentation'
    )

    filename_zarr = output_zarr_store / f'/{gene_id}/{grna_id}/{grna_trench_id}'
    # with open_ome_zarr(filename_zarr, mode='r+') as store:
    #     store["mKate"] = fl_image
        # store["1"] = seg_image
    print(filename_zarr)
    break

