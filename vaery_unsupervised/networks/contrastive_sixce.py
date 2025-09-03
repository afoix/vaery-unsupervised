import logging
from typing import Literal

import torch
import torch.nn.functional as F
from monai.networks.nets.resnet import ResNetFeatures
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
from torch import Tensor, nn
from typing_extensions import TypedDict
from pytorch_lightning import LightningModule


_logger = logging.getLogger("lightning.pytorch")

class ContrastiveSample(TypedDict):
    """
    Triplet sample type for mini-batches.
    """

    anchor: Tensor
    positive: Tensor

def projection_mlp(in_dims: int, hidden_dims: int, out_dims: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dims, hidden_dims),
        nn.BatchNorm1d(hidden_dims),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dims, out_dims),
        nn.BatchNorm1d(out_dims),
    )

class ResNetEncoder(nn.Module):
    def __init__(self, backbone: str,
        in_channels: int = 147,
        spatial_dims: int = 2, #3
        embedding_dim: int = 512,
        mlp_hidden_dims: int = 768,
        projection_dim: int = 128,
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

        self.encoder = ResNetFeatures(self.backbone, pretrained=self.pretrained, spatial_dims=self.spatial_dims, in_channels=self.in_channels)
        self.projection = projection_mlp(in_dims=self.embedding_dim,
                                         hidden_dims=self.mlp_hidden_dims,
                                         out_dims=self.projection_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
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
        feature_map = self.encoder(x)[-1]
        embedding = self.encoder.avgpool(feature_map)
        embedding = embedding.view(embedding.size(0), -1)
        projections = self.projection(embedding)
        return (embedding, projections)


class ContrastiveModule(LightningModule):
    def __init__(self, encoder: nn.Module | ResNetEncoder, lr: float = 1e-3,
                 optimizer=None, temperature: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        self.optimizer = optimizer

        base_loss = NTXentLoss(temperature=temperature)
        self.loss = SelfSupervisedLoss(base_loss)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Return the embedding and the projection
        """
        return self.encoder(x)
    
    def on_train_start(self):
        _logger.debug(f"Training started with {self.encoder.backbone} backbone")
        super().on_train_start()

        hparams = { #FIXME pass dict of hyperparams from main
            # Training hyperparameters
            "lr": self.lr,
            # "schedule": self.schedule,
            "input_shape": self.example_input_array, 
            "loss_function_class": self.loss.__class__.__name__,
        }
        # Tensorboard Logger hyperparameters
        if self.logger is not None:
            self.logger.log_hyperparams(hparams)
            

    def on_validation_start(self):
        _logger.debug(f"Validation started with {self.encoder.backbone} backbone")
        super().on_validation_start()

    def training_step(self, batch: ContrastiveSample, batch_idx: int) -> Tensor:
        anchor, positive = batch["anchor"], batch["positive"]
        _, anchor_proj = self(anchor)
        _, positive_proj = self(positive)
        loss = self.loss(anchor_proj, positive_proj)  # NTXentLoss
        self._log_metrics(loss, anchor_proj, positive_proj, "train")
        return loss

    def configure_optimizers(self):
        if self.optimizer:
            return self.optimizer(self.parameters(), lr=self.lr)
        else:
            return torch.optim.AdamW(self.parameters(), lr=self.lr) # mkw changed from Adam

    def validation_step(self, batch: ContrastiveSample, batch_idx: int) -> Tensor:
        anchor, positive = batch["anchor"], batch["positive"]
        _, anchor_proj = self(anchor)
        _,positive_proj = self(positive)

        #Compute the loss with the projections pairs
        loss = self.loss(anchor_proj, positive_proj)

        # NOTE: Use our convenience function to log the metrics otherwise use just self.log()
        self._log_metrics(loss, anchor_proj, positive_proj, "val")
        return loss

    def predict_step(self, batch, batch_idx):
        images = batch["anchor"]
        cell_ids = batch["cell_id"]

        embedding, projection = self(images)

        return {
            'embeddings': embedding,
            'projections': projection,
            'cell_ids': cell_ids,
            'batch_idx': batch_idx
        }

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

        # Compute cosine similarity and euclidian distance for positive pairs
        cosine_sim_pos = F.cosine_similarity(anchor, positive, dim=1).mean()
        euclidean_dist_pos = F.pairwise_distance(anchor, positive).mean()

        log_metric_dict = {
            f"metrics/cosine_similarity_positive/{stage}": cosine_sim_pos,
            f"metrics/euclidean_distance_positive/{stage}": euclidean_dist_pos,
        }
        # lightning logger to tensorboard (at epoch level)
        self.log_dict(
            log_metric_dict,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def _log_images(self, anchor: Tensor, positive: Tensor, stage: Literal["train", "val"]):
        NotImplementedError("Logging images is not implemented")