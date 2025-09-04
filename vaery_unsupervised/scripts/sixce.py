import os
import numpy as np
import pandas as pd
import torch
import zarr
import sys
import pytorch_lightning as pl
from lightning.pytorch.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings

# Local imports
from vaery_unsupervised.dataloaders.dataloader_sixce import SixceDataModule, inspect_pt_file
from vaery_unsupervised.networks.contrastive_sixce import ResNetEncoder, ContrastiveModule
from vaery_unsupervised.scripts.utils import run_inference



# Mode & Description ---------------------------------------------------------------------------------------------------
INSPECT_DATASET_AND_EXIT = False
pickle_to_inspect = "sixce_dataset_2kCells.pt"

RUN_INFERENCE_ONLY = True
checkpoint_to_load = "SIXCE-20/epoch=29-step=3330.ckpt" # str: run_id/checkpoint_file_name.ckpt"

OVERWRITE_DATASET = False

# Neptune experiment description
EXPERIMENT_DESCRIPTION = (f"SIXCE: resnet34, mlp 768, proj 128, 2k-cells (tau 0.25)")
NEPTUNE_TOKEN_PATH = "/dgx1nas1/storage/data/kamal/neptune_token"
with open(NEPTUNE_TOKEN_PATH, "r") as f:
    api_token = f.read().strip()


# Paths ----------------------------------------------------------------------------------------------------------------
DATA_PATH = '/dgx1nas1/storage/data/kamal/sixce_data'
DATAFRAMES_PATH = os.path.join(DATA_PATH, 'dataframes')
SAVED_DATA_PATH = os.path.join(DATA_PATH, "saved_datasets")
OME_ZARR_PATH = '/raid/data/temp_kamal/merged_mosaics.ome.zarr'
CHECKPOINT_PATH = os.path.join(DATA_PATH, "checkpoints")
SAVED_INFERENCE_PATH = os.path.join(DATA_PATH, "inferences")

# Training dataset parameters (params overwritten if loading existing dataset) -----------------------------------------
RANDOM_SUBSET = 2000 # if -1, uses all samples
# Unsorted list of stains to include in the training. Will get sorted downstream based on the OME-Zarr attributes
STAINS_unsorted = ['DAPI', 'Cellbound1', 'Cellbound2', 'Cellbound3', 'Ki67', 'WT1', 'PolyT']
N_MASKS = 0 # cell mask (numeric bool)
GENE_COUNT_THRESHOLD = -1
APPLY_DATA_AUGMENTATION = True
TENSOR_TYPE = torch.float32
INPUT_DIM = 200 #206  # crop size
LAZY_LOAD = True
SPOT_REPRESENTATION_TYPE = 'psf' # point spread function 'psf' or 'binary'
SPOT_PSF_GAUSSIAN_SIGMA = 1.0 # sigma of gaussian patch applied in psf representation
SPOT_PSF_GAUSSIAN_KERNEL_SIZE = 9 # size of gaussian kernel in psf representation (side of square or diameter of circle)
SPOT_PSF_PX_VAL_NORMALIZATION_METHOD = 'peak' # gaussian patch pixel val norm: peak, sum or sum_corrected
PEAK_NORMALIZATION_PEAK_VALUE = 0.8 # Value of peak if using the peak normalization method for PSF
CIRCULAR_PSF = True # If false, PSFs are square-truncated
PADDING = 4 # spot padding for binary spot representation

# Training parameters --------------------------------------------------------------------------------------------------
GLOBAL_SEED = 137
LEARNING_RATE = 0.0005
EPOCHS = 50
BATCH_SIZE = 72 #vRAM=31.7GiB@80, 30.7GiB@72
N_WORKERS = 10 # 10 workers, pre-fetch=2, batch sz=72 costs 250GiB vRAM
VALIDATION_FRACTION=0.2
ENCODER_BACKBONE = 'resnet34' # available: resnet18, 34, 50, 101, 152, 200
MLP_HIDDEN_DIMS = 768 #256
PROJECTION_DIMS = 128

# Loss function parameters ---------------------------------------------------------------------------------------------
LOSS_TEMPERATURE = 0.25

# Experiment variables -------------------------------------------------------------------------------------------------
DRY_RUN = False
EPOCHS = 1 if DRY_RUN else EPOCHS

# Loading pickled dataset for inspection and exiting
if INSPECT_DATASET_AND_EXIT:
    print("Loading existing Dataset for inspection...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        _pt_file = torch.load(os.path.join(SAVED_DATA_PATH, pickle_to_inspect))
        inspect_pt_file(_pt_file, max_items=20, max_repr=50)
        sys.exit(0)

# Constructing file name string based on RANDOM_SUBSET
if RANDOM_SUBSET == -1:
    subset_size_str = "all"
elif RANDOM_SUBSET >= 1000:
    subset_size_str = f"{RANDOM_SUBSET // 1000}k"
else:
    subset_size_str = f"{RANDOM_SUBSET / 1000:.5f}".lstrip('0').rstrip('0').rstrip('.').replace('.', '0_') + 'k'

expected_filename = f"sixce_dataset_{subset_size_str}Cells.pt"
saved_dataset_path = os.path.join(
    SAVED_DATA_PATH,
    expected_filename
)

# Preparing data
# Load existing dataset if possible
if os.path.exists(saved_dataset_path) and not OVERWRITE_DATASET:
    print("Loading existing Dataset...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        dataset = torch.load(saved_dataset_path)

    # Extracting stored variables
    N_GENES = dataset.n_genes
    INPUT_DIM = dataset.input_dim
    STAINS = dataset.selected_stains
    N_STAINS = len(STAINS)
    # N_MASKS = len(train_dataset.stain_order) - N_STAINS
    N_MASKS = dataset.n_masks  # mkw
    STAIN_TO_INDEX = {s: i for i, s in enumerate(dataset.stain_order) if s != "mask"}
    APPLY_DATA_AUGMENTATION = dataset.apply_data_augmentation
    TENSOR_TYPE = dataset.tensor_type
    REPRESENTATION_TYPE = dataset.representation_type
    SPOT_PSF_GAUSSIAN_SIGMA = dataset.gaussian_patch_sigma
    SPOT_PSF_GAUSSIAN_KERNEL_SIZE = dataset.gaussian_kernel_size
    PSF_NORMALIZATION = dataset.psf_normalization
    RANDOM_SUBSET = dataset.random_subset

    # Initializing None vars
    zarr_attributes=None
    cell_metadata_df=None
    cell_boundaries_df=None

    # Loading gene expression df for inference
    print("Loading detected_transcripts.csv")
    # Load dataframe with the detected transcripts
    transcript_df_path = os.path.join(DATAFRAMES_PATH, "detected_transcripts.csv")
    transcript_df = pd.read_csv(transcript_df_path)
    # Removing blank probes/rounds
    blank_rows = transcript_df["gene"].str.startswith("Blank-")
    transcript_df.drop(index=transcript_df.index[blank_rows], inplace=True)

else:
    dataset=None
    # Loading OME-Zarr metadata
    print("Loading OME-Zarr metadata")
    zarr_attributes = zarr.open(f"{OME_ZARR_PATH}/0", mode="r").attrs

    # Sorting the STAINS_unsorted list and indexing as per OME-Zarr, same as the Dataset class
    ome_channels = [ch["label"] for ch in zarr_attributes["omero"]["channels"]]
    STAINS = [label for label in ome_channels
              if label in STAINS_unsorted]
    STAIN_TO_INDEX = {stain: i for i, stain in enumerate(STAINS)}
    # Guard against typos
    missing = set(STAINS_unsorted) - set(STAINS)
    if missing:
        raise ValueError(f"These stains are not in the OME-Zarr: {missing}")
    N_STAINS = len(STAINS)

    print("Loading detected_transcripts.csv")
    # Load dataframe with the detected transcripts
    transcript_df_path = os.path.join(DATAFRAMES_PATH, "detected_transcripts.csv")
    transcript_df = pd.read_csv(transcript_df_path)

    # Removing blank probes/rounds
    blank_rows = transcript_df["gene"].str.startswith("Blank-")
    transcript_df.drop(index=transcript_df.index[blank_rows], inplace=True)

    # Apply gene count thresholding
    if GENE_COUNT_THRESHOLD != -1:
        print("Applying gene filtering")
        # Keep only barcode_ids that occur more than the threshold
        barcode_counts = transcript_df['barcode_id'].value_counts()
        valid_barcodes = barcode_counts[barcode_counts > GENE_COUNT_THRESHOLD].index
        transcript_df = transcript_df[transcript_df['barcode_id'].isin(valid_barcodes)]

    # Re-indexing (barcode_id) in-place based on gene name order (strings)
    uniq_genes = np.sort(transcript_df["gene"].unique())
    gene_to_id = {g: i for i, g in enumerate(uniq_genes)}

    N_GENES = len(uniq_genes)
    if N_GENES <= 255:
        id_dtype = np.uint8
    else:  # In case of a bigger dataset
        id_dtype = np.uint16
    transcript_df["barcode_id"] = transcript_df["gene"].map(gene_to_id).astype(id_dtype)

    print("Loading cell_metadata.csv")
    # Load cell metadata df
    cell_metadata_df_path = os.path.join(DATAFRAMES_PATH, "cell_metadata.csv")
    cell_metadata_df = pd.read_csv(cell_metadata_df_path)

    print("Loading cell_boundaries.parquet")
    # Load cell boundaries df
    cell_boundaries_df_path = os.path.join(DATAFRAMES_PATH, "cell_boundaries.parquet")
    cell_boundaries_df = pd.read_parquet(cell_boundaries_df_path)

datamodule_config = {
            "data_path": OME_ZARR_PATH,
            "dataset_export_path": saved_dataset_path,
            "batch_size": BATCH_SIZE,
            "prefetch_factor": 2,
            "pin_memory": False,
            "persistent_workers": False,
            "n_workers": N_WORKERS,
            "val_fraction": VALIDATION_FRACTION,
            "zarr_attributes": zarr_attributes,
            "transcripts_df": transcript_df,
            "cell_metadata_df": cell_metadata_df,
            "cell_boundaries_df": cell_boundaries_df,
            "input_dim": INPUT_DIM,
            "n_genes": N_GENES,
            "existing_dataset": dataset,
            "selected_stains": STAINS,
            "n_masks": N_MASKS,
            "padding": PADDING,
            "apply_data_augmentation": APPLY_DATA_AUGMENTATION,
            "tensor_type": TENSOR_TYPE,
            "include_stains": not LAZY_LOAD,
            "representation_type": SPOT_REPRESENTATION_TYPE,
            "gaussian_patch_sigma": SPOT_PSF_GAUSSIAN_SIGMA,
            "gaussian_kernel_size": SPOT_PSF_GAUSSIAN_KERNEL_SIZE,
            "psf_normalization": SPOT_PSF_PX_VAL_NORMALIZATION_METHOD,
            "random_subset": RANDOM_SUBSET,
            "peak_norm_peak_val": PEAK_NORMALIZATION_PEAK_VALUE,
            "circular_psf": CIRCULAR_PSF,
            "global_seed": GLOBAL_SEED
}


logged_dataset_config = {
            **datamodule_config,
            "gene_threshold": GENE_COUNT_THRESHOLD,
            "tensor_type": str(TENSOR_TYPE),
            "stains": ",".join(STAINS)
        }

# Remove irrelevant entries
keys2del = ('zarr_attributes', 'transcripts_df', 'cell_metadata_df', 'cell_boundaries_df', 'existing_dataset',
            'selected_stains')
for k in keys2del:
    logged_dataset_config.pop(k, None)

# Setting embedding dim according to encoder backbone (fixed arch)
if ENCODER_BACKBONE in ["resnet18", "resnet34"]:
    embedding_dim = 512
elif ENCODER_BACKBONE in ["resnet50", "resnet101", "resnet152", "resnet200"]:
    embedding_dim = 2048
else:
    raise ValueError(f"ResNet backbone unsupported: {ENCODER_BACKBONE}")


def main():
    pl.seed_everything(GLOBAL_SEED, workers=True)

    dm = SixceDataModule(**datamodule_config)

    encoder = ResNetEncoder(
        backbone=ENCODER_BACKBONE,
        in_channels=N_GENES + N_STAINS,
        embedding_dim=embedding_dim,
        mlp_hidden_dims=MLP_HIDDEN_DIMS,
        projection_dim=PROJECTION_DIMS
    )

    if not RUN_INFERENCE_ONLY:
        model = ContrastiveModule(encoder=encoder, lr=LEARNING_RATE, optimizer=None, temperature=LOSS_TEMPERATURE)

        # Initialize Neptune
        if DRY_RUN:
            run_id = "dry_run"
        else:
            logger = NeptuneLogger(
                api_key=api_token,
                project="BroadImagingPlatform/SIXCE",
                tags=["training", "resnet18"],
                log_model_checkpoints=False,
            )

            neptune_run = logger.experiment
            neptune_run["Experiment"] = EXPERIMENT_DESCRIPTION
            run_id = neptune_run["sys/id"].fetch()
            neptune_run["dataset_config"] = logged_dataset_config

        # Local checkpoints
        current_checkpoint_path = os.path.join(CHECKPOINT_PATH, run_id)
        os.mkdir(current_checkpoint_path)

        trainer = pl.Trainer(
            accelerator="gpu",
            num_nodes=1,
            # precision="16-mixed",
            strategy="auto",
            max_epochs=EPOCHS,
            logger=logger,
            devices=1,
            # fast_dev_run=True, #Only when debugging
            callbacks = [ModelCheckpoint( dirpath=current_checkpoint_path,
                filename="{epoch}-{step}", save_last=True, monitor="loss/val", save_top_k=8, every_n_epochs=1)],
        )

        # Train
        dm.setup(stage="fit")
        trainer.fit(model, datamodule=dm)

        # Predict (returns and auto saves embeddings, projections, cell_ids, and annotated PCA, UMAP, and tSNE)
        run_inference(
            saved_inference_path=SAVED_INFERENCE_PATH,
            model_name=run_id,
            data_module=dm,
            model=model,
            global_seed=GLOBAL_SEED,
            transcript_df=transcript_df,
            annotation_marker_list=-1 # None for no annotations, -1 for all genes
        )

    elif checkpoint_to_load is not None:
        model_path = os.path.join(CHECKPOINT_PATH, checkpoint_to_load)
        model = ContrastiveModule.load_from_checkpoint(model_path, encoder=encoder)
        model_name = checkpoint_to_load.split('/')[0]

        # Sanity check embedding by annotating based on expression of some marker genes
        # marker_list = ["AQP2", "PODXL", "PLA2R1", "MZB1", "NKG7", "C1QA", "C1QB", "C1QC", "FCN1", "TRAC", "FCGR3A",
        #                "PTPRC", "CLDN5", "ESM1", "SPARCL1"]

        run_inference(
            saved_inference_path=SAVED_INFERENCE_PATH,
            model_name=model_name,
            data_module=dm,
            model=model,
            global_seed=GLOBAL_SEED,
            transcript_df=transcript_df,
            annotation_marker_list=None
        )


if __name__ == "__main__":
    main()

