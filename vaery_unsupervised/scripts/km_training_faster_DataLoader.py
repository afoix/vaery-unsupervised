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
from vaery_unsupervised.networks.LitVAE_km import SpatialVAE
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

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
lightning_module = SpatProtoZarrDataModule(
    out_path/"converted_crops_with_metadata.zarr",
    masking_function=simple_masking,
    dataset_normalisation_dict=DATASET_NORM_DICT,
    transform_both=transform_both,
    transform_input=transform_input,
    num_workers=8,
    batch_size=16,
)
lightning_module.setup("train")

loader = lightning_module.train_dataloader()
# %% looking at a batch
for i,batch in enumerate(loader):
    plot_batch_sample(batch)    
    break
# %%
batch["input"][:,[1,2,3],:,:].shape

spatialmodel = SpatialVAE(n_chan=3,latent_size=128, lr = 0.001, beta = 0.01)
spatialmodel(batch["input"][:,[1,2,3],:,:])[0].shape

#%%
logging_path = Path("/mnt/efs/aimbl_2025/student_data/S-KM/logs")
logging_path.mkdir(exist_ok=True)


logger = TensorBoardLogger(save_dir=logging_path, name = "test_fasterloader")

def main(*args, **kwargs):

    trainer = lightning.Trainer(
        max_epochs = 1, 
        accelerator = "gpu", 
        precision = "16-mixed", 
        logger=logger,
        callbacks=[ModelCheckpoint(save_last=True,save_top_k=8,monitor='val/loss',every_n_epochs=1),],
        log_every_n_steps=5
    )
    #callback = Callback.on_save_checkpoint(trainer = trainer, pl_module = lightning_module, checkpoint = )


    # run training and validation
    trainer.fit(model = spatialmodel , datamodule = lightning_module)
    trainer.validate(model = spatialmodel , datamodule = lightning_module)

#%%
if __name__ == "__main__":
   main()

