#%% testing
from vaery_unsupervised.dataloaders.dataloader_km_ryans_template import (
    DATASET_NORM_DICT,
    SpatProteomicDataModule,
    SpatProteomicsDataset,
    simple_masking,
    DataLoader
)
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

sdata = dataset.datasets[0]
mean = sdata["czi_450"].mean(dim = ['y','x'])
std = sdata["czi_450"].std(dim = ['y','x'])
# %%
# transforms = [affine, flips, shear, gauss_noise, smoothing, intensity, variation]
# %%
for i in range(600)[::20]:
    print(dataset[i]["raw"].mean(axis=(1,2)))

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
    train_val_split=0.9,
    polygon_sdata_name='affine_transformed',
    seed=42,
    train=True,
)
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
# %%
loader = lightningmodule.train_dataloader()
#%%
for batch in loader:
    tes_batch = batch
    break
tes_batch

# %%

plot_batch_sample(tes_batch)
# %%
