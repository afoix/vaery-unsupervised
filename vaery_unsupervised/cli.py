import lightning

from vaery_unsupervised.dataloaders.MNISTDataModule import MNISTDataModule
from vaery_unsupervised.networks.LitVAE import LitVAE


def main(*args, **kwargs):
  litmodel = LitVAE()
  datamod = MNISTDataModule()

  trainer = lightning.Trainer(max_epochs = 1)

  # run training and validation
  trainer.fit(model = litmodel, datamodule = datamod)
  trainer.validate(model = litmodel, datamodule = datamod)
