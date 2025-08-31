#%%
"""
Conversion from CZI to OME-Zarr

Examples:
https://czbiohub-sf.github.io/iohub/main/auto_examples/run_multi_fov_hcs_ome_zarr.html#sphx-glr-auto-examples-run-multi-fov-hcs-ome-zarr-py

For Nikon or other supported file formats:
https://github.com/glencoesoftware/bioformats2raw?tab=readme-ov-file#usage
"""
from iohub import open_ome_zarr
from tifffile import tifffile
import numpy as np


output_zarr_store = '/output/dataset.zarr'
path_to_dataset = ''
tifffile.imread("test.czi")

# This creates a zarr store
with open_ome_zarr(output_zarr_store, mode='w') as store:
    # TODO: this encodes for /row/column/fov 
    row = 0
    column = 0
    fov = 0
    position = store.create_position(row,column,fov)

    # TODO: replace for the array. All the arrays should be the same shape
    position["0"] = np.random.randint(
        0, np.iinfo(np.uint16).max, size=(5, 3, 2, 32, 32), dtype=np.uint16
    )

#%%
from iohub.ngff.utils import create_empty_plate

output_zarr_store = '/output/dataset.zarr'

#TODO: position keys for the HCS dataset format.
position_keys = [(row,column,fov),...]

T,C,Z,Y,X = array.shape #TODO: get the shape of the array to be converted. (T,C,Z,Y,X)

#TODO: 
create_empty_plate(store_path=output_zarr_store,
    position_keys = position_keys,
    channel_names = ['channel_1'], #TODO: replace for the channel names
    shape = (T,C,Z,Y,X), #TODO: replace for the shape (T,C,Z,Y,X)
    chunks = None, #TODO: this is important for training (T,C,Z,Y,X)
    scale = (1, 1, 1, 1, 1), #TODO: [Optional] replace for the scale in um. (T,C,Z,Y,X)
    dtype= np.float32, 
)

# Parallel processing
from concurrent.futures import ProcessPoolExecutor

def convert_file(file_path:str, position_keys:tuple):
    # Logic to open a file and fill in the plate
    image = tifffile.imread(file_path)
    with open_ome_zarr(output_zarr_store+f'/{position_keys[0]}/{position_keys[1]}/{position_keys[2]}', mode='r+') as store:
        store["0"] = image

file_paths = ['file_path1', 'file_path2', 'file_path3']
with ProcessPoolExecutor(max_workers=10) as executor:
    # TODO: make the matching files with the position keys
    file_paths = [file_path for file_path in file_paths]
    position_keys = [position_key for position_key in position_keys]
    executor.map(convert_file, file_paths, position_keys)
