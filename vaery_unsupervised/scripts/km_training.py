#%%
from vaery_unsupervised.dataloaders.dataloader_km_ryans_template import *
from vaery_unsupervised.km_utils import *
import monai.transforms as transforms

# %%
patient_id_list = ["450","453","456_2","457","493_2", "543_2"] 
prefix = "/mnt/efs/aimbl_2025/student_data/S-KM/001_Used_Zarrs/onlybatch1/"
suffix = "_sdata_at"
data_paths = [prefix + patient + suffix for patient in patient_id_list]
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
                                           batch_size=60,
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

#%%
loader = lightning_module.train_dataloader()

# %%
for batch in loader:
   batch = batch
   break
# %%
batch["input"][:,[1,2,3],:,:].shape
#%% 
from vaery_unsupervised.networks.LitVAE_km import SpatialVAE
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback

# %%
spatialmodel = SpatialVAE(n_chan=3,latent_size=128, lr = 0.001, beta = 0.001)
spatialmodel(batch["input"][:,[1,2,3],:,:])[0].shape
#%%
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

logging_path = Path("/mnt/efs/aimbl_2025/student_data/S-KM/logs")
logging_path.mkdir(exist_ok=True)


logger = TensorBoardLogger(save_dir=logging_path, name = "vae_3")

def main(*args, **kwargs):

    trainer = lightning.Trainer(max_epochs = 100, accelerator = "gpu", precision = "16-mixed", logger=logger,
                                callbacks=[ModelCheckpoint(save_last=True,save_top_k=8,monitor='val/loss',every_n_epochs=1),]
                                )
    #callback = Callback.on_save_checkpoint(trainer = trainer, pl_module = lightning_module, checkpoint = )


    # run training and validation
    trainer.fit(model = spatialmodel , datamodule = lightning_module)
    trainer.validate(model = spatialmodel , datamodule = lightning_module)





#%%
main()
# %% good practice when running python file
if __name__ == "__main__":
   main()