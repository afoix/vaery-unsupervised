#%% testing
from vaery_unsupervised.dataloaders.dataloader_km_ryans_template import (
    DATASET_NORM_DICT,
    SpatProteomicDataModule,
    SpatProteomicsDataset,
    simple_masking,
    DataLoader,
    polygon_to_centered_mask
)
from vaery_unsupervised.dataloaders.dataloader_km_ryans_template import transpose_polygon_to_image
import rasterio.features
from vaery_unsupervised.km_utils import (plot_batch_sample, plot_dataloader_output)
import numpy as np
import monai.transforms as transforms

prefix = "/mnt/efs/aimbl_2025/student_data/S-KM/001_Used_Zarrs/onlybatch1/"
suffix = "_sdata_at"
name_list = ["450","453","456_2","457","493_2", "543_2"] 
file_list = [prefix + name + suffix for name in name_list]
file_list

# means_stddev_dict = {}
# for data_name in name_list:
#     file_name = prefix + data_name + suffix 
#     sdata = sd.read_zarr(file_name)
#     mean, stddev = get_crop_mean_stdev(sdata,data_name)
#     means_stddev_dict[data_name] = (mean, stddev)
#%%
means_stddev_dict = DATASET_NORM_DICT
#%%

dataset = SpatProteomicsDataset(
    file_list=file_list,
    patient_id_list=name_list,
    masking_function=simple_masking,
    dataset_normalisation_dict=means_stddev_dict,
    transform_input=None,
    transform_both = None,
    train_val_split=0.9,
    polygon_sdata_name='affine_transformed',
    seed=42,
    train=True,
)
#%%
len(dataset)

# %%
for i in range(600)[::20]:
    print(dataset[i]["raw"].mean(axis=(1,2)))
#%%
timing = dataset[0]["timing"]
timing
#%%
import spatialdata as sd
import time
file_name = prefix + name_list[0] + suffix 
sdata = sd.read_zarr(file_name)
sdata
#%% # check timing of getting an image
idx = 0
name_image = f'czi_{name_list[0]}'
name_coordinates = 'faligned{name_list[0]}'
name_transformed_poly = 'affine_transformed'
size=128

start = time.time_ns()
subset_sdata = sdata.subset([name_coordinates,name_image,name_transformed_poly]) #subset sdata object to keep coordinates, the image, as well as the transformed
poly = subset_sdata[name_transformed_poly].iloc[idx]['geometry'] #the polygon subsets the gdf and gives you the geometry from that idx
well = subset_sdata[name_transformed_poly].index[idx] #this gives you the plate id for that cell, ie the actual shape
centroid = poly.centroid.coords[0] #gives you the centroid of the polygon
time_getting_poly_n_subset = time.time_ns() - start

start = time.time_ns()
cropped = subset_sdata.query.bounding_box(
    axes = ['x', 'y'],
    min_coordinate=(np.array(centroid)-size/2),
    max_coordinate=(np.array(centroid)+size/2),
    target_coordinate_system=name_coordinates
)
time_getting_bbox = time.time_ns() - start

start = time.time_ns()
mask= polygon_to_centered_mask(poly,size)
time_getting_mask = time.time_ns() - start

# -> seems that the sdata step is the problem
print(time_getting_poly_n_subset/1000000000)
print(time_getting_bbox/1000000000)
print(time_getting_mask/1000000000)
# %%
transform_both = [
    transforms.RandAffine(prob=0.5, 
        rotate_range=3.14, 
        shear_range=(0,0,0), 
        translate_range=(0,20,20), 
        scale_range=None,   
        padding_mode="zeros",
        spatial_size=(128,128)),
    transforms.RandFlip(prob = 0.5,
                                spatial_axis = [-1], 
                            ),]

transform_input = [
    transforms.RandGaussianNoise(
        prob = 0.5,
        mean = 0,
        std = 1
    ),
]
# %%
transformed_dataset = SpatProteomicsDataset(
    file_list=file_list,
    patient_id_list=name_list,
    masking_function=simple_masking,
    dataset_normalisation_dict=means_stddev_dict,
    transform_input=transform_input,
    transform_both = transform_both,
    train_val_split=1,
    polygon_sdata_name='affine_transformed',
    seed=42,
    train=True,
)
#%% go through data and check timing / if any bugs appear
import pandas as pd
out_times = []
for i in range(0,len(transformed_dataset)):
    print(i)
    out_time = pd.DataFrame(transformed_dataset[i]["timing"], index = [i])
    out_times.append(out_time)

#%%
pd.concat(out_times).mean()
# %%
for i in np.random.randint(0, len(transformed_dataset), 10):
    x = transformed_dataset[i]
    plot_dataloader_output(x)

# %%
lightningmodule = SpatProteomicDataModule(
    data_paths=file_list,
    patient_id_list=name_list,
    masking_function=simple_masking,
    dataset_normalisation_dict=means_stddev_dict,
    transform_input=transform_input,
    transform_both = transform_both,
    polygon_sdata_name='affine_transformed',
     num_workers=1,
     batch_size=4,
#     prefetch_factor=None,
#     pin_memory=False,
)
lightningmodule.setup("train")
# %%
loader = DataLoader(
    lightningmodule.dataset,
    batch_size=4,
    shuffle = False
)
for batch in loader:
    tes_batch = batch
    break
#%%
tes_batch["raw"].shape
#%%

#%%
loader = lightningmodule.train_dataloader()
#%%
for batch in loader:
    tes_batch = batch
    break
tes_batch

# %%

plot_batch_sample(tes_batch)
# %%
