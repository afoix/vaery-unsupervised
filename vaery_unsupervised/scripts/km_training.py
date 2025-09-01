#%%
from vaery_unsupervised.dataloaders.dataloader_km_ryans_template import *
from vaery_unsupervised.km_utils import *
import monai.transforms as transforms

# %%
patient_id_list = ["450","453","456_2","457","493_2", "543_2"] 
prefix = "/mnt/efs/aimbl_2025/student_data/S-KM/001_Used_Zarrs/onlybatch1/"
suffix = "_sdata_at"
data_paths = [prefix + patient + suffix for patient in patient_id_list ]
DATASET_NORM_DICT

#%%
transform_both = [
    transforms.RandAffine(
        prob = 0.5,
        rotate_range=(3.14, 0,0),
        shear_range=(0,0,0),
        translate_range=(0,20,20),
        padding_mode="zeros"), 
    transforms.RandFlip(prob = 0.5,
                        spatial_axis=[-1])
                         ] #flips along y axis


transform_input = [
    transforms.RandGaussianNoise(
        prob = 0.5,
        mean = 0,
        std = 0.5
    )
]

#%%
lightning_module = SpatProteomicDataModule(patient_id_list= patient_id_list,
                                           data_paths=data_paths, 
                                           polygon_sdata_name="affine_transformed",
                                           masking_function=simple_masking,
                                           batch_size=10,
                                           num_workers=1,
                                           crop_size = 128,
                                           dataset_normalisation_dict= DATASET_NORM_DICT,
                                           transform_both=transform_both,
                                           transform_input=transform_input,
                                           seed = 42,
                                           train_val_split=0.9
                                           )

# %%
lightning_module.setup("train")
# %%
loader = lightning_module.train_dataloader()
#%%
plot_dataloader_output(loader.dataset[0])
# %%
for batch in loader:
    tes_batch = batch
    break
tes_batch
# %%

# %%
for idx in range(len(loader.dataset)):
    print(idx)
    out = loader.dataset[idx]
# %%
loader.dataset[239]
# %%
