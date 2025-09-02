#!/usr/bin/env python
from torch import Tensor
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from vaery_unsupervised.networks.LitVAE import LitVAE
from vaery_unsupervised.dataloaders.sal_brain_loader import SalBrainDataModule

def main():

    dataset = SalBrainDataModule(batch_size=9, 
                                patch_size=(32, 32, 32), 
                                num_workers=64)

    dataset.setup("train")
    train_data = dataset.train_dataloader()

    sample = next(iter(train_data))
    batch_shape = sample.shape

    model = LitVAE(beta=1e-3, 
                   matrix_size=32,
                   latent_size=128, 
                   n_chan=batch_shape[1])
    
    trainer = L.Trainer(accelerator="gpu", 
                        max_epochs=10)
    
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = Trainer(logger=logger)
    
    trainer.fit(model=model, 
                train_dataloaders=train_data)
    

    




if __name__ == "__main__":
    main()


