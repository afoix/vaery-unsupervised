#!/usr/bin/env python
import lightning as L
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint  # , LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from vaery_unsupervised.dataloaders.sal_brain_loader import SalBrainDataModule
from vaery_unsupervised.networks.SalamanderVAE import SalamanderVAE

seed_everything(1010)


def main():
    #torch.set_float32_matmul_precision("high")

    model_name = "sal_model_v2_z1024_b32_e8_sz64"

    dataset = SalBrainDataModule(
        batch_size=32,
        patch_size=(64, 64, 64),
        num_workers=96,
        pin_memory=True,
        persistent_workers=True,
    )

    dataset.setup("fit")
    train_data = dataset.train_dataloader()
    val_data = dataset.validation_dataloader()

    sample = next(iter(train_data))
    batch_shape = sample.shape
    print(f"Batch shape: {batch_shape}")

    model = SalamanderVAE(
        beta=1e-8,
        matrix_size=batch_shape[3],
        latent_size=1024,
        n_chan=batch_shape[1],
        z_dir=f"/home/jnc2161/mbl/{model_name}_latent",
    )

    logger_tb = TensorBoardLogger(save_dir="/home/jnc2161/mbl/logs", name=model_name)
    trainer = L.Trainer(
        accelerator="gpu",
        # precision='16-mixed',
        max_epochs=1000,
        check_val_every_n_epoch=5,
        logger=logger_tb,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(save_top_k=10, monitor="val/loss", every_n_epochs=5)
        ],
    )

    trainer.fit(model=model, 
                train_dataloaders=train_data, 
                val_dataloaders=val_data) 
                #ckpt_path=f"/home/jnc2161/mbl/logs/{model_name}/version_0/checkpoints/epoch=4-step=250.ckpt")


if __name__ == "__main__":
    main()
