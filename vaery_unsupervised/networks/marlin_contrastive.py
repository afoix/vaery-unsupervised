
import logging
from typing import Literal

import torch
import torch.nn.functional as F
from lightning import LightningModule
from monai.networks.nets.resnet import ResNetFeatures
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
from torch import Tensor, nn
from typing_extensions import TypedDict
from vaery_unsupervised.networks.marlin_utils import detach_sample, render_images

_logger = logging.getLogger("lightning.pytorch")

class ContrastiveSample(TypedDict):
    """
    Triplet sample type for mini-batches.
    """

    anchor: Tensor
    positive: Tensor

def projection_mlp(
        in_dims: int,
        hidden_dims: int,
        out_dims: int
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dims, hidden_dims),
        nn.BatchNorm1d(hidden_dims),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dims, out_dims),
        nn.BatchNorm1d(out_dims),
    )

class ResNetEncoder(nn.Module):
    def __init__(
        self, 
        backbone: str,
        in_channels: int = 1,
        spatial_dims: int = 2,#2,
        embedding_dim: int = 512,#768,
        mlp_hidden_dims: int = 768,
        projection_dim: int = 32,
        pretrained: bool = False,
    ):
        """
        ResNetEncoder modified with the projection MLP layer for contrastive learning

        Available backbones checkout TIMM:
        - resnet18, resnet34, resnet50, resnet101, resnet152, resnet200

        Parameters
        ----------
        backbone : str
            The backbone to use
        in_channels : int
            The number of input channels
        embedding_dim : int
            The dimension of the embedding
        projection_dim : int
            The dimension of the projection
        """
        super().__init__()
        self.backbone = backbone
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels

        self.embedding_dim = embedding_dim  
        self.mlp_hidden_dims = mlp_hidden_dims
        self.projection_dim = projection_dim

        self.pretrained = pretrained

        if self.backbone not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet200"]:
            raise ValueError(f"Backbone {self.backbone} not supported")

        self.resnet = ResNetFeatures(
            self.backbone,
            pretrained=self.pretrained,
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels
        )

        self.projection = projection_mlp(
            in_dims=self.embedding_dim,
            hidden_dims=self.mlp_hidden_dims,
            out_dims=self.projection_dim
        )

    def forward(
            self,
            x: Tensor
        ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input image

        Returns
        -------
        tuple[Tensor, Tensor]
            The embedding tensor and the projection tensor
        """
        # feature_map = self.encoder(x)[-1]
        feature_map = self.resnet(x)[-1]
        # embedding = self.encoder.avgpool(feature_map)
        embedding = self.resnet.avgpool(feature_map)
        embedding = embedding.view(embedding.size(0), -1)
        projections = self.projection(embedding)
        return (embedding, projections)

class ContrastiveModule(LightningModule):
    def __init__(
        self,
        encoder: nn.Module| ResNetEncoder,
        loss: nn.Module| NTXentLoss, 
        lr: float = 1e-3,
        optimizer = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.loss = loss
        self.lr = lr
        self.optimizer = optimizer
    def forward(
            self,
            x: Tensor
        ) -> tuple[Tensor, Tensor]:
        """
        Return the embedding and the projection
        """
        return self.encoder(x)
    
    def on_train_start(self):
        # _logger.debug(f"Training started with {self.encoder.backbone} backbone")
        _logger.debug(f"Training started with {self.encoder} backbone")
        self.train_examples = {'anchor': [], 'positive': []}
        self.val_examples = {'anchor': [], 'positive': []}
        super().on_train_start()

        # 
        hparams = {
            # Training hyperparameters
            "lr": self.lr,
            # "schedule": self.schedule,
            # "input_shape": self.example_input_array, 
            "loss_function_class": self.loss.__class__.__name__,
        }
        # Tensorboard Logger hyperparameters
        if self.logger is not None:
            self.logger.log_hyperparams(hparams)
            
    def on_validation_start(self):
        # _logger.debug(f"Validation started with {self.encoder.backbone} backbone")
        _logger.debug(f"Validation started with {self.encoder} backbone")
        super().on_validation_start()

    def training_step(
        self,
        batch: ContrastiveSample,
        batch_idx: int
    ) -> Tensor:
        anchor, positive = batch["anchor"], batch["positive"]
        # Get the embedding and the projection
        _, anchor_proj = self(anchor)
        _, positive_proj = self(positive)

        #Compute the loss with the projections pairs
        #FIXME: evaluate the use of the SelfSupervisedLoss and the NTXentLoss
        if isinstance(self.loss, SelfSupervisedLoss):
            loss = self.loss(anchor_proj, positive_proj)
        else:
            # Throw error
            raise ValueError("Unsupported loss function")
            # if isinstance(self.loss, NTXentLoss):
            #     indices = torch.arange(anchor_proj.size(0))
                
            #     print()
            #     loss = self.loss(anchor_proj, positive_proj)#, indices)
            # else:
            #     loss = self.loss(anchor_proj, positive_proj)

        # NOTE: Use our convenience function to log the metrics otherwise use just self.log()
        if batch_idx < 4:
            self.train_examples['anchor'].append(batch['anchor'])
            self.train_examples['positive'].append(batch['positive'])
        self._log_metrics(loss, anchor, positive, "train")
        return loss

    def configure_optimizers(self):
        # TODO: expose this as an argument to the constructor so we can use different optimizers
        if self.optimizer:
            return self.optimizer(self.parameters(), lr=self.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def validation_step(
        self,
        batch: ContrastiveSample, batch_idx: int
    ) -> Tensor:
        anchor, positive = batch["anchor"], batch["positive"]
        _, anchor_proj = self(anchor)
        _,positive_proj = self(positive)
        
        #Compute the loss with the projections pairs
        loss = self.loss(anchor_proj, positive_proj)
        
        # NOTE: Use our convenience function to log the metrics otherwise use just self.log()
        self._log_metrics(loss = loss, anchor= anchor_proj,  positive=positive_proj, stage= "val")
        if batch_idx < 4:
            self.val_examples['anchor'].append(batch['anchor'])
            self.val_examples['positive'].append(batch['positive'])
        return loss
    
    def predict_step(
        self,
        batch
    ):
        anchor, positive = batch["anchor"], batch["positive"]
        anchor_emb, anchor_proj = self(anchor)
        # positive_emb, positive_proj = self(positive)

        #Compute the loss with the projections pairs

        return anchor_emb, anchor_proj

    def on_validation_epoch_end(self):
        _logger.debug(f"Validation epoch ended with {self.encoder.backbone} backbone")
        self._log_images(self.val_examples['anchor'], "val")
        self._log_images(self.val_examples['positive'], "val")
        self.val_examples.clear()
        super().on_validation_epoch_end()


    def on_training_epoch_end(self):
        _logger.debug(f"Training epoch ended with {self.encoder.backbone} backbone")
        self._log_images(self.train_examples['anchor'], "train")
        self._log_images(self.train_examples['positive'], "train")
        self.train_examples.clear()
        super().on_training_epoch_end()

    def _log_metrics(
        self, loss, anchor, positive, stage: Literal["train", "val"]
    ):
        self.log(
            f"loss/{stage}",
            loss.to(self.device),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Compute codsine similarity and euclidian distance for positive pairs
        cosine_sim_pos = F.cosine_similarity(anchor, positive, dim=1).mean()
        euclidean_dist_pos = F.pairwise_distance(anchor, positive).mean()

        log_metric_dict = {
            f"metrics/cosine_similarity_positive/{stage}": cosine_sim_pos,
            f"metrics/euclidean_distance_positive/{stage}": euclidean_dist_pos,
        }
        # lightning logger to tensorboard (at epoch level)
        self.log_dict(
            log_metric_dict,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def _log_images(self, examples, stage, anchor_or_positive: Literal["anchor", "positive"] = "anchor"):
        example_numpy = detach_sample(examples,log_samples_per_batch=8)
        image = render_images(example_numpy)
        self.logger.experiment.add_image(
            f"{stage}/{anchor_or_positive}_examples",
            image,
            global_step=self.current_epoch,
            dataformats="HWC",
        )