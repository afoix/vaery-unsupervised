#%%
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import yaml
import argparse
import torch
import pickle
import numpy as np

from vaery_unsupervised.dataloaders.marlin_dataloader.marlin_dataloader import MarlinDataModule
from vaery_unsupervised.networks.marlin_contrastive import ContrastiveModule
from vaery_unsupervised.networks.marlin_custom_resnet import SmallObjectResNet10Encoder
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
from monai.transforms import (
    Compose,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
)

# Map transform names from config to actual classes
TRANSFORM_MAP = {
    "RandFlipd": RandFlipd,
    "RandGaussianNoised": RandGaussianNoised,
    "RandGaussianSmoothd": RandGaussianSmoothd,
}

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_transforms(transforms_config):
    """Instantiates a MONAI Compose transform from a list of configurations."""
    transform_list = []
    for t_config in transforms_config:
        name = t_config.pop('name')
        transform_class = TRANSFORM_MAP.get(name)
        if transform_class:
            transform_list.append(transform_class(**t_config))
        else:
            raise ValueError(f"Unknown transform: {name}")
    return Compose(transform_list)

def count_total_embeddings_in_files(embedding_files, metadata_length):
    """
    Count the total number of embeddings across all embedding files and compare with metadata length.
    """
    total_embeddings = 0
    for embedding_file in embedding_files:
        with open(embedding_file, "rb") as f:
            embeddings_dict = pickle.load(f)
            batch_size = embeddings_dict['embeddings'][0].shape[0]
            total_embeddings += batch_size

    print(f"Total embeddings found: {total_embeddings}, Metadata length: {metadata_length}")
    assert total_embeddings == metadata_length, "Mismatch between total embeddings and metadata length!"

def generate_embedding_and_projection_matrices(embeddings_directory, metadata_path):
    """
    Generate embedding and projection matrices from saved embedding files.
    """
    embedding_files = sorted(glob.glob(str(embeddings_directory / 'embeddings_batch_*.pkl')))
    if not embedding_files:
        print("No embedding files found. Please run prediction first.")
        return None, None

    # Load first file to determine the shape of the embedding and projection matrices
    with open(embedding_files[0], "rb") as f:
        first_batch = pickle.load(f)
    embedding_dim = first_batch['embeddings'].shape[1]
    
    metadata = pd.read_pickle(metadata_path)
    current_index = 0
    
    # Initialize the full matrices
    embedding_matrix = np.zeros((len(metadata), embedding_dim))
    
    # Fill the matrices
    for embedding_file in embedding_files:
        print(f'Loading embeddings from {embedding_file}')
        with open(embedding_file, "rb") as f:
            embeddings_dict = pickle.load(f)
            embedding = embeddings_dict['embeddings'].cpu().numpy()
            
            len_batch = embedding.shape[0]
            start_index = current_index
            end_index = start_index + len_batch
            
            embedding_matrix[start_index:end_index] = embedding
            current_index += len_batch
    
    count_total_embeddings_in_files(embedding_files, len(metadata))
    
    return embedding_matrix, metadata


def main(config_path, checkpoint_path, mode):
    """Main function to perform embedding prediction using a config file."""
    print("Loading configuration from:", config_path)
    config = load_config(config_path)
    
    # Load paths from config
    paths_config = config['paths']
    head_path = Path(paths_config['head_path'])
    metadata_path = head_path / paths_config['metadata_balanced_path']
    data_path = head_path / paths_config['data_path']

    # Instantiate the encoder and model from config
    encoder_params = config['encoder']
    marlin_encoder = SmallObjectResNet10Encoder(**encoder_params)

     # Instantiate the ContrastiveModule from config
    contrastive_config = config['contrastive_module']
    loss = SelfSupervisedLoss(NTXentLoss(temperature=contrastive_config['loss']['temperature']))

    # Load the model from the checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    try:
        model = ContrastiveModule.load_from_checkpoint(
            checkpoint_path,
            encoder=marlin_encoder,
            loss=loss,
            lr=contrastive_config['lr'],
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Please ensure the checkpoint path is correct and the model architecture matches.")
        return

    # Set model to evaluation mode
    model.eval()

    # Instantiate the data transforms from config
    transforms = get_transforms(config['transforms'])

    # Instantiate the DataModule from config
    data_module_config = config['data_module']
    data_module = MarlinDataModule(
        data_path=data_path,
        metadata_path=metadata_path,
        split_ratio=data_module_config['split_ratio'],
        batch_size=data_module_config['batch_size'],
        num_workers=data_module_config['num_workers'],
        prefetch_factor=data_module_config['prefetch_factor'],
        transforms=transforms,
    )

    # Set up the data module for prediction
    data_module.setup(stage='predict')
    
    # Define output path and create directory
    ### NEED MORE CONFIG
    embeddings_directory = head_path / paths_config['embeddings_directory']
    embeddings_directory.mkdir(parents=True, exist_ok=True)


    # Instantiate the Trainer for prediction
    trainer_params = config['trainer']
    trainer = L.Trainer(
        devices=trainer_params['devices'],
        accelerator=trainer_params['accelerator'],
        strategy=trainer_params['strategy'],
        precision=trainer_params['precision'],
        max_epochs=trainer_params['max_epochs'],
        fast_dev_run=trainer_params['fast_dev_run'],
    )
    
    # Perform prediction
    print("Starting prediction...")


    if mode == 'single':
        trainer = L.Trainer(
            devices=config['trainer']['devices'],
            accelerator=config['trainer']['accelerator'],
            strategy=config['trainer']['strategy'],
            precision=config['trainer']['precision'],
            max_epochs=config['trainer']['max_epochs'],
            fast_dev_run=config['trainer']['fast_dev_run'],
        )
        embeddings_list = trainer.predict(
            model=model,
            dataloaders=data_module.predict_dataloader(),
            return_predictions=True
        )
        
        # Concatenate all embeddings from the list of tensors
        embeddings = torch.cat(embeddings_list, dim=0).cpu().numpy()
        
        print(f"Prediction complete. Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")
    
        embeddings_path = embeddings_directory / 'embeddings_single.pkl'
        print(f"Saving embeddings to {embeddings_path}")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
    elif mode == 'batch':
        dataloader = data_module.predict_dataloader()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                embeddings_batch = model(batch['anchor'].to('cuda' if torch.cuda.is_available() else 'cpu'))
                filename = embeddings_directory / f'embeddings_batch_{i:06d}.pkl'
                print(f'Saving embeddings for batch {i} to {filename}')
                with open(filename, "wb") as f:
                    pickle.dump({'id': batch['id'], 'embeddings': embeddings_batch}, f)
        
        # After saving, re-assemble the matrices for use
        # NOTE: this part requires enough memory to hold the full matrices
        print("Batch-wise saving complete. Assembling full matrices...")
        embedding_matrix, metadata = generate_embedding_and_projection_matrices(embeddings_directory, metadata_path)
        
        # You can now use embedding_matrix and metadata for further analysis
        if embedding_matrix is not None:
            print("Full embedding matrix and metadata loaded successfully.")
            # Example: Save the full matrix for later use
            full_matrix_path = embeddings_directory / 'embeddings_full_matrix.pkl'
            with open(full_matrix_path, 'wb') as f:
                pickle.dump({'embeddings': embedding_matrix, 'metadata': metadata}, f)
            print(f"Full embedding matrix saved to {full_matrix_path}")

    else:
        raise ValueError(f"Unknown mode: {mode}. Please use 'single' or 'batch'.")

    print("Embeddings process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings using a trained model and a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file (e.g., /path/to/last.ckpt).")
    parser.add_argument("--mode", type=str, choices=['single', 'batch'], default='single', help="Prediction mode: 'single' for full-dataset prediction, 'batch' for batch-wise saving.")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.mode)

%%
%load_ext autoreload
%autoreload 2
import pathlib
from lightning import LightningModule
from pathlib import Path
import torch.nn.functional as F
import torch
from vaery_unsupervised.networks.marlin_contrastive import ContrastiveModule, ResNetEncoder
from monai.networks.nets.resnet import ResNetFeatures
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss

import torch

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from vaery_unsupervised.dataloaders.marlin_dataloader.marlin_dataloader import MarlinDataModule, MarlinDataset
from vaery_unsupervised.networks.marlin_contrastive import ResNetEncoder, ContrastiveModule
from vaery_unsupervised.networks.marlin_resnet import SmallObjectResNet10Encoder
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
import pickle
from vaery_unsupervised.dataloaders.marlin_dataloader.marlin_dataloader import MarlinDataModule
from sklearn.decomposition import PCA
from monai.transforms import (
    Compose,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd
)
import pathlib
import glob

transforms = Compose([
    # NormalizeIntensity(
    #     subtrahend=MEAN_OVER_DATASET,
    #     divisor=STD_OVER_DATASET,
    #     nonzero=False,
    #     channel_wise=False,
    #     dtype = np.float32,
    # ),
    RandFlipd(
        keys=["positive"],
        spatial_axis=0,
        prob=0.5,
    ),
    RandFlipd(
        keys=["positive"],
        spatial_axis=1,
        prob=0.5,
    ),
    RandGaussianNoised(
        keys=["positive"],
        prob=0.5,
        mean=0,
        std=0.5,
        sample_std=True,
    ),
    
    RandGaussianSmoothd(
        keys=["positive"],
        sigma_x=(0.5, 1.5),
        sigma_y=(0.5, 1.5),
        prob=0.5
    )
])

# Get the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGS_HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/tb_logs/marlin_contrastive_custom_resnet/')
checkpoint_path = LOGS_HEADPATH / 'version_1/checkpoints/last.ckpt'

marlin_encoder = SmallObjectResNet10Encoder(
        in_channels=1,
        widths=(32, 64, 128, 256),   # slim to reduce overfitting
        feature_stage="layer4",      # use "layer3" if you want even fewer params
        layer4_dilate=3,             # captures long rods without further downsampling
        norm="batch", gn_groups=8,   # better than BN for small batches
        drop_path_rate=0.05,         # mild stochastic depth
        mlp_hidden_dims=512,         # set = embedding_dim if you prefer
        projection_dim=128,
    )

marlin_contrastive_config = {
        "encoder": marlin_encoder,
        "loss": SelfSupervisedLoss(NTXentLoss(temperature=0.07)),
        "lr": 1e-3,#1e-3,
    }

marlin_contrastive = ContrastiveModule(**marlin_contrastive_config)

loss = SelfSupervisedLoss(NTXentLoss(temperature=0.07))
lr = 1e-3
optimizer = torch.optim.Adam(marlin_encoder.parameters(), lr=lr)

model = ContrastiveModule.load_from_checkpoint(
    checkpoint_path,
    encoder=marlin_encoder,
    loss=loss,
    lr=lr,
    optimizer=optimizer,
)
model.to(device)
model.eval()

HEADPATH = Path('/mnt/efs/aimbl_2025/student_data/S-GL/')
METADATA_PATH = HEADPATH / '2025-08-31_lDE20_Final_Barcodes_df_Merged_Clustering_expanded.pkl'
METADATA_BALANCED_PATH  = HEADPATH / '2025-09-03_lDE20_Final_balanced_30tr-per-gene.pkl'
EMBEDDINGS_DIRECTORY = HEADPATH / 'embeddings_30tr-per-gene_all-t_v21/'
# EMBEDDINGS_PATH = HEADPATH / 'embeddings_30tr-per-gene_all-t_v21.pkl'

data_module = MarlinDataModule(
    data_path=HEADPATH/'Ecoli_lDE20_Exps-0-1/',
    metadata_path=METADATA_BALANCED_PATH,
    split_ratio=0.8,
    batch_size=512,#16,#64,
    num_workers=8,
    prefetch_factor=2,
    transforms=transforms,
)
data_module.setup(stage='predict')

## Do single step if all the embeddings can fit in memory, otherwise do batch-wise prediction and save to disk
DO_SINGLE_STEP = True
if DO_SINGLE_STEP:
    trainer = L.Trainer(
            devices="auto",
            accelerator="gpu",
            strategy="auto",
            precision="16-mixed",
            max_epochs=1000,
            fast_dev_run=False,
            # logger=logger,
            log_every_n_steps=5,
            callbacks=[
                ModelCheckpoint(
                    monitor="loss/val",
                    save_top_k=8,
                    every_n_epochs=1,
                    save_last=True
                )
            ],
        )
    
    embeddings = trainer.predict(
        model=model,
        dataloaders=data_module.predict_dataloader(),
        return_predictions=True
    )
else:
    dataloader = data_module.predict_dataloader()
    with torch.no_grad():
        pathlib.Path(EMBEDDINGS_DIRECTORY).mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(dataloader):
            embeddings_batch = model(batch['anchor'].to(device))
            filename = EMBEDDINGS_DIRECTORY / f'embeddings_batch_{i:06d}.pkl'
            print(f'Saving embeddings for batch {i} to {filename}')
            with open(filename, "wb") as f:
                pickle.dump({'id': batch['id'], 'embeddings': embeddings_batch}, f)

def generate_embedding_and_projection_matrices():
    """
    Generate embedding and projection matrices from saved embedding files.
    """
    embedding_files = sorted(glob.glob(str(EMBEDDINGS_DIRECTORY / 'embeddings_batch_*.pkl')))
    # Load first file to determine the shape of the embedding and projection matrices
    with open(embedding_files[0], "rb") as f:
        embeddings = pickle.load(f)
    embedding_dim = embeddings['embeddings'][0].shape[1]
    projection_dim = embeddings['embeddings'][1].shape[1]
    metadata = pd.read_pickle(METADATA_BALANCED_PATH)
    embedding_matrix = np.zeros((len(metadata),embedding_dim))
    projection_matrix = np.zeros((len(metadata), projection_dim))

    for embedding_file in embedding_files:
        print(f'Loading embeddings from {embedding_file}')
        with open(embedding_file, "rb") as f:
            embeddings = pickle.load(f)
            embedding = embeddings['embeddings'][0].cpu().numpy()
            projection = embeddings['embeddings'][1].cpu().numpy()
            len_batch = embedding.shape[0]
            start_index = current_index
            end_index = start_index + len_batch
            embedding_matrix[start_index:end_index] = embedding
            projection_matrix[start_index:end_index] = projection
            current_index += len_batch

def count_total_embeddings_in_files(embedding_files, metadata):
    """
    Count the total number of embeddings across all embedding files and compare with metadata length.
    """
    total_embeddings = 0
    for embedding_file in embedding_files:
        with open(embedding_file, "rb") as f:
            embeddings = pickle.load(f)
            batch_size = embeddings['embeddings'][0].shape[0]
            total_embeddings += batch_size

    print(f"Total embeddings: {total_embeddings}, Metadata length: {len(metadata)}")
    assert total_embeddings == len(metadata), "Mismatch between total embeddings and metadata length!"

# def random_subset_embeddings_and_metadata(
#     embedding_matrix,
#     projection_matrix,
#     metadata,
#     n_samples=1000
# ):
#     total_samples = embedding_matrix.shape[0]
#     subset_indices = np.random.choice(total_samples, n_samples, replace=False)
#     subset_embeddings = embedding_matrix[subset_indices]
#     subset_projection = projection_matrix[subset_indices]
#     subset_metadata = metadata.iloc[subset_indices]
#     return subset_embeddings, subset_projection, subset_metadata

# def subset_late_timepoints_embeddings_and_metadata(
#     embedding_matrix,
#     projection_matrix,
#     metadata,
#     timepoint_threshold=45
# ):
#     late_timepoint_mask = metadata['timepoints'] > timepoint_threshold
#     subset_embeddings = embedding_matrix[late_timepoint_mask]
#     subset_projection = projection_matrix[late_timepoint_mask]
#     subset_metadata = metadata[late_timepoint_mask]
#     return subset_embeddings, subset_projection, subset_metadata

# from sklearn.decomposition import PCA
# def perform_pca_on_embeddings(embedding_matrix):
#     pca = PCA(n_components=2)
#     pcs = pca.fit_transform(embedding_matrix)
#     return pcs
