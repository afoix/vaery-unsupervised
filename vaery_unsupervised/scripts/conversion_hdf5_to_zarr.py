#%%
from pathlib import Path
from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
import h5py
import pandas as pd
import numpy as np

# HEADPATH = Path('/home/lag36/scratch/lag36/Ecoli/2023-01-18_lDE20_Merged_Analysis/') # HMS Cluster
HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/') # AI-MBL Cluster
metadata_path = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering.pkl'
output_zarr_store = HEADPATH / 'Ecoli_lDE20_Exps-0-1_v2.zarr'
path_to_dataset = HEADPATH / 'Ecoli_lDE20_Exps-0-1'

def load_array_from_hdf5(
    file_idx: int,
    headpath: Path,
    prefix: str,
    key: str) -> np.ndarray:
        '''
        Load a 2D array from a specific key in an HDF5 file.
        '''
        filepath = headpath / f"{prefix}{file_idx}.hdf5"
        with h5py.File(filepath, 'r') as f:
            images = f[key][:]
        return images

# Get position keys from metadata
metadata = pd.read_pickle(metadata_path)
position_keys = tuple(
    (str(gene_id), str(oDEPool7_id), str(grna_file_trench_index))
    for gene_id, oDEPool7_id, grna_file_trench_index in metadata.loc[:, ['gene_id', 'oDEPool7_id', 'grna_file_trench_index']].itertuples(index=False, name=None)
)

# Get shape from first image
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
# print(zarr_array_shape)
# Create Zarr store and transfer files
with open_ome_zarr(
    output_zarr_store,
    layout="hcs",
    mode="w-",
    channel_names=['mKate'],
) as dataset:
     for gene_id, grna_id, grna_trench_id in position_keys:
        # Show nicely formated time
        position = dataset.create_position(gene_id, grna_id, grna_trench_id)
        # position['0'] = np.empty(
        #         shape = zarr_array_shape,
        #         dtype= np.float32,
        # )
        # position['labels'] = np.empty(
        #     shape = zarr_array_shape,
        #     dtype= np.float32,
        # )
        fl_image = load_array_from_hdf5(
            file_idx=f'{grna_id}',
            headpath=path_to_dataset,
            prefix=f'{gene_id}/',
            key='fluorescence'
        )[int(grna_trench_id)]

        seg_image = load_array_from_hdf5(
            file_idx=f'{grna_id}',
            headpath=path_to_dataset,
            prefix=f'{gene_id}/',
            key='segmentation'
        )[int(grna_trench_id)]
        # print(fl_image.shape)
        position['0'] = fl_image[:, None, None, :, :]
        position['labels'] = seg_image[:, None, None, :, :]
        #  dataset["mKate"] = fl_image
        #  dataset["1"] = seg_image

#%% Print Zarr store info
# from iohub.reader import print_info
# print_info(output_zarr_store, verbose=True)
# # dataset = open_ome_zarr(data_path)

# #%% Create Zarr store
# PIXEL_SIZE_MICRONS = 0.105951961895293*2
# create_empty_plate(store_path=output_zarr_store,
#     position_keys = position_keys,
#     channel_names = ['mkate', 'labels'], #TODO: replace for the channel names
#     shape = zarr_array_shape, #TODO: replace for the shape (T,C,Z,Y,X)
#     chunks = zarr_array_shape, #TODO: this is important for training (T,C,Z,Y,X)
#     scale = (1, 1, 1, PIXEL_SIZE_MICRONS, PIXEL_SIZE_MICRONS), #TODO: [Optional] replace for the scale in um. (T,C,Z,Y,X)
#     dtype= np.float32,#np.float32, 
# )
# #%%
# def convert_hdf5_to_zarr_trench(
#           headpath_hdf5,
#           headpath_zarr,
#           gene_id,
#           grna_id,
#           grna_trench_id,
# ):
#     fl_image = load_array_from_hdf5(
#         file_idx=f'{grna_id}',
#         headpath=headpath_hdf5,
#         prefix=f'{gene_id}/',
#         key='fluorescence'
#     )[int(grna_trench_id)]

#     seg_image = load_array_from_hdf5(
#         file_idx=f'{grna_id}',
#         headpath=headpath_hdf5,
#         prefix=f'{gene_id}/',
#         key='segmentation'
#     )[int(grna_trench_id)]
#     print(headpath_zarr)
#     filename_zarr = str(output_zarr_store) + f'/{gene_id}/{grna_id}/{grna_trench_id}'
#     print(filename_zarr)
#     print(fl_image.shape, seg_image.shape)
#     # with open_ome_zarr(filename_zarr, mode='r+') as store:
#     #     store["mKate"] = fl_image
#     #     store["1"] = seg_image


# #%%
# convert_hdf5_to_zarr_trench(
#     headpath_hdf5=path_to_dataset,
#     headpath_zarr=output_zarr_store,
#     gene_id='3',
#     grna_id='14193',
#     grna_trench_id='0'
# )

# #%% Parallel processing
# # Parallelize by gene
# position_keys
# #%%
# for i, (gene_id, grna_id, grna_trench_id) in enumerate(position_keys):
#     convert_hdf5_to_zarr_trench(
#         headpath_hdf5=path_to_dataset,
#         headpath_zarr=output_zarr_store,
#         gene_id=gene_id,
#         grna_id=grna_id,
#         grna_trench_id=grna_trench_id
#     )
#     if i == 3:
#         break

# #%% Parallel processing
# from concurrent.futures import ProcessPoolExecutor
# with ProcessPoolExecutor(max_workers=10) as executor:
#     for i, (gene_id, grna_id, grna_trench_id) in enumerate(position_keys):
#         executor.map(
#             convert_hdf5_to_zarr_trench,
#             headpath_hdf5=path_to_dataset,
#             headpath_zarr=output_zarr_store,
#             gene_id=gene_id,
#             grna_id=grna_id,
#             grna_trench_id=grna_trench_id
#         )
#     i