#%% testing
import numpy as np
from pathlib import Path
from vaery_unsupervised.dataloaders.dataloader_km_ryans_template import (
    simple_masking, 
    DATASET_NORM_DICT, 
    SpatProteoDatasetZarr,
    SpatProtoZarrDataModule,
)
from vaery_unsupervised.km_utils import plot_batch_sample,plot_dataloader_output
import monai.transforms as transforms
out_path = Path("/mnt/efs/aimbl_2025/student_data/S-KM/")

transform_both = [
    transforms.RandAffine(
        prob=0.5, 
        rotate_range=3.14, 
        shear_range=(0,0,0), 
        translate_range=(0,20,20), 
        scale_range=None,   
        padding_mode="zeros",
        spatial_size=(128,128)),
    transforms.RandFlip(
        prob = 0.5,
        spatial_axis = [-1], 
    ),
]
transform_input = [
    transforms.RandGaussianNoise(
        prob = 0.5,
        mean = 0,
        std = 1
    ),
]

dataset_zarr = SpatProteoDatasetZarr(
    out_path/"converted_crops_with_metadata.zarr",
    masking_function=simple_masking,
    dataset_normalisation_dict=DATASET_NORM_DICT,
    transform_both=transform_both,
    transform_input=transform_input
)
plot_dataloader_output(dataset_zarr[0])
# %%
lightning = SpatProtoZarrDataModule(
    out_path/"converted_crops_with_metadata.zarr",
    masking_function=simple_masking,
    dataset_normalisation_dict=DATASET_NORM_DICT,
    transform_both=transform_both,
    transform_input=transform_input,
    num_workers=8,
    batch_size=4,
)
lightning.setup("train")
loader = lightning.train_dataloader()
# %% looking at a batch
for i,batch in enumerate(loader):
    plot_batch_sample(batch)    
    break
# %%
