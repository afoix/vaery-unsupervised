import lightning

from vaery_unsupervised.dataloaders.dataloader_km_ryans_template import SpatProteomicDataModule
from vaery_unsupervised.networks import LightningVAE_linear_km 


def main(*args, **kwargs):
  litmodel = LitVAE()
  datamod = SpatProteomicDataModule()

  trainer = lightning.Trainer(max_epochs = 1000)

  # run training and validation
  trainer.fit(model = litmodel, datamodule = datamod)
  trainer.validate(model = litmodel, datamodule = datamod)
