#%%

from pathlib import Path

import numpy as np
from tifffile import tifffile
import torch
from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import create_empty_plate
from glob import glob
from natsort import natsorted
import re
from collections import defaultdict


def parse_filename(filename):
    """
    Parse HCS filename format like 'r07c09f02p01-ch5sk1fk1fl1.tiff'
    
    Parameters
    ----------
    filename : str or Path
        Filename to parse
        
    Returns
    -------
    dict
        Dictionary with parsed components: row, col, field, position, channel, etc.
    """
    filename_str = Path(filename).name
    
    # Pattern to match r##c##f##p##-ch#sk#fk#fl#.tiff
    pattern = r'r(\d{2})c(\d{2})f(\d{2})p(\d{2})-ch(\d+)sk(\d+)fk(\d+)fl(\d+)\.tiff'
    match = re.match(pattern, filename_str)
    
    if not match:
        raise ValueError(f"Filename {filename} does not match expected format")
    
    return {
        'row': int(match.group(1)),
        'col': int(match.group(2)),
        'field': int(match.group(3)),
        'position': int(match.group(4)),
        'channel': int(match.group(5)),
        'sk': int(match.group(6)),
        'fk': int(match.group(7)),
        'fl': int(match.group(8))
    }
#%%
# Inpput data path and storing all the tiffs into a list
input_data_path = '/mnt/efs/aimbl_2025/student_data/S-RM/full_dataset/Images'
file_paths = [Path(path) for path in natsorted(glob(f"{input_data_path}/*.tiff"))]

# Store the regex match to the parsed dict
test_regex = parse_filename(file_paths[0])
print(f'test: {test_regex}')
parsed = [parse_filename(fp) for fp in file_paths]
print(f'parsed: {parsed}')

# Create the position keys
position_keys = []
for value in parsed:
    position_keys.append((value['row'], value['col'], value['field'],value['channel']))
print(f'position_keys: {position_keys}')

#%%
# Debug and check that the positions and regex match for 10 positions
for i, (key, file_path) in enumerate(zip(position_keys, file_paths) ):
    print(f'key: {key}, file_path: {file_path}')
    if i == 10:
        break

#%%
# Open a test tiff to check the shape
img = tifffile.imread(file_paths[0])
Y,X = img.shape
print(img.shape)
    
#%%
# output_zarr_path = Path('/mnt/efs/aimbl_2025/student_data/S-RM/full_dataset/test_output.zarr')

# previous_position_keys = None
# with open_ome_zarr(output_zarr_path, mode="w") as store:
#     for position_keys, file in zip(position_keys, file_paths):
#         # TODO: make a position that is unique (row, col, field)
#         position = store.create_position(row,col,field)
#         # TODO: Load the tiff and make sure it is a 5D array (t,c,z,y,x) in our case (1,3,1,1080,1080)
#         stack_tcyx = tczyx
#         position.create_image('0', czyx_array)


output_zarr_path = Path('/mnt/efs/aimbl_2025/student_data/S-RM/full_dataset/RM_project_ome.zarr')
target_channels = [1, 4, 5]

position_groups = defaultdict(list)
for file_path in file_paths:
    parsed = parse_filename(file_path)
    position_key = (parsed['row'], parsed['col'], parsed['field'])
    position_groups[position_key].append({
        'file_path': file_path,
        'channel': parsed['channel']
    })

with open_ome_zarr(output_zarr_path, mode="w", layout ='hcs', channel_names = ['mito','er','nuclei']) as store:
    for (row, col, field), files in position_groups.items():
        # Create channel to file mapping
        channel_to_file = {f['channel']: f['file_path'] for f in files}
        
        # Load and stack target channels
        channel_images = []
        for channel in target_channels:
            img = tifffile.imread(channel_to_file[channel])
            channel_images.append(img)
        
        # Create 5D array (t, c, z, y, x)
        stacked = np.stack(channel_images, axis=0,dtype=np.float32)
        stack_tczyx = stacked[np.newaxis, :, np.newaxis, :, :]
        
        # Save to zarr
        position = store.create_position(row, col, field)
        position.create_image('0', stack_tczyx,)


#%%
import matplotlib.pyplot as plt
# read the data

with open_ome_zarr(output_zarr_path/'1'/'1'/'1', mode="r") as store:
    img = store[0][0,2,0,]
    plt.imshow(img, cmap='grey')
# %%
