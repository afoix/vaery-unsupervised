from typing import Any
from pathlib import Path
from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples
from microsplit_reproducibility.datasets.custom_dataset_2D import load_one_file
from microsplit_reproducibility.utils.paper_metrics import avg_range_inv_psnr, compute_SE, _get_list_of_images_from_gt_pred
from microssim import MicroMS3IM, MicroSSIM
from numpy.typing import NDArray
from skimage.measure import pearson_corr_coeff
from skimage.metrics import structural_similarity
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def _normalize_for_lpips(imgs: list[NDArray]) -> NDArray:
    """Normalize the given image in [0, 1] for LPIPS.
    
    Parameters
    ----------
    img : NDArray
        A list of multi-channels images to normalize, each one of shape (C, Z, Y, X).
    
    Returns
    -------
    NDArray
        The normalized image.
    """
    # TODO: use training dset stats for normalization (?)
    ax_idxs = tuple(range(1, imgs[0].ndim))
    min_ = np.min([img.min(axis=ax_idxs) for img in imgs])
    max_ = np.max([img.max(axis=ax_idxs) for img in imgs])
    min_ = np.asarray(min_).reshape(-1, *np.ones_like(ax_idxs, dtype=int))
    max_ = np.asarray(max_).reshape(-1, *np.ones_like(ax_idxs, dtype=int))
    return np.array([(img - min_) / (max_ - min_) for img in imgs])


def lpips(
    prediction: Union[np.ndarray, torch.Tensor], 
    target: Union[np.ndarray, torch.Tensor]
) -> float:
    """Compute the Learned Perceptual Image Patch Similarity (LPIPS) over images.
    
    If inputs are 3D, LPIPS is averaged over the Z-stack.
    
    NOTES:
    - LPIPS can use different networks. Here we use the SqueezeNet model.
    - The inputs are expected to be normalized in the range [0, 1].
    - We use the mean reduction, i.e., the LPIPS value is averaged over the batch.

    Parameters
    ----------
    prediction : Union[np.ndarray, torch.Tensor]
        Array of predicted images, shape is (N, C, [Z], Y, X).
    target : Union[np.ndarray, torch.Tensor]
        Array of ground truth images, shape is (N, C, [Z], Y, X).

    Returns
    -------
    float
        LPIPS value over the batch.
    """
    assert prediction.shape == target.shape, "Prediction and target shapes must match."
    assert prediction.max() <= 1 and prediction.min() >= 0, (
        "Prediction must be normalized in [0, 1]."
    )
    assert target.max() <= 1 and target.min() >= 0, (
        "Target must be normalized in [0, 1]."
    )
    
    # check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # compute LPIPS
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='squeeze', reduction='mean', normalize=True
    ).to(device)
    
    if len(prediction.shape) == 5: # 3D input
        # iterate over Z-stack
        return np.mean([
            lpips(
                torch.tensor(prediction[:, :, i], device=device, dtype=torch.float32),
                torch.tensor(target[:, :, i], device=device, dtype=torch.float32)
            ).item()
            for i in range(prediction.shape[2])
        ])
    else:
        return lpips(
            torch.tensor(prediction, device=device, dtype=torch.float32),
            torch.tensor(target, device=device, dtype=torch.float32)
        ).item()


def compute_metrics(
    gt_data,
    pred_unnorm,
    metrics,
):
    """
    last dimension is the channel dimension
    """
    psnr_list = []
    pearson_list = []
    microssim_list = []
    ms3im_list = []
    ssim_list = []
    msssim_list = []
    lpips_list = []
    for ch_idx in range(gt_data[0].shape[-1]):
        # list of gt and prediction images. This handles both 2D and 3D data. 
        # This also handles when individual images are lists.
        gt_ch, pred_ch = _get_list_of_images_from_gt_pred(
            gt_data, pred_unnorm, ch_idx
        )
        
        # PSNR
        if "PSNR" in metrics:
            psnr_list.append(avg_range_inv_psnr(gt_ch, pred_ch))

        # MicroSSIM
        if "MicroSSIM" in metrics:
            microssim_obj = MicroSSIM()
            microssim_obj.fit(gt_ch, pred_ch)
            mssim_scores = [
                microssim_obj.score(gt_ch[i], pred_ch[i]) for i in range(len(gt_ch))
            ]
            microssim_list.append((np.mean(mssim_scores), compute_SE(mssim_scores)))

        # MicroS3IM
        if "MicroMS3IM" in metrics:
            m3sim_obj = MicroMS3IM()
            m3sim_obj.fit(gt_ch, pred_ch)
            ms3im_scores = [
                m3sim_obj.score(gt_ch[i], pred_ch[i]) for i in range(len(gt_ch))
            ]
            ms3im_list.append((np.mean(ms3im_scores), compute_SE(ms3im_scores)))
        
        # SSIM
        if "SSIM" in metrics:
            ssim = [
                structural_similarity(
                    gt_ch[i], pred_ch[i], data_range=gt_ch[i].max() - gt_ch[i].min()
                )
                for i in range(len(gt_ch))
            ]
            ssim_list.append((np.mean(ssim), compute_SE(ssim)))

        # MSSSIM
        if "MS-SSIM" in metrics:
            ms_ssim = []
            for i in range(len(gt_ch)):
                ms_ssim_obj = MultiScaleStructuralSimilarityIndexMeasure(
                    data_range=gt_ch[i].max() - gt_ch[i].min()
                )
                ms_ssim.append(
                    ms_ssim_obj(
                        torch.Tensor(pred_ch[i][None, None]),
                        torch.Tensor(gt_ch[i][None, None]),
                    ).item()
                )
            msssim_list.append((np.mean(ms_ssim), compute_SE(ms_ssim)))

        # Pearson's Correlation Coefficient
        if "Pearson" in metrics:
            pearson_scores = [
                pearson_corr_coeff(gt_ch[i], pred_ch[i])
                for i in range(len(gt_ch))
            ]
            pearson_list.append((np.mean(pearson_scores), compute_SE(pearson_scores)))
            
        # LPIPS
        if "LPIPS" in metrics:
            lpips_scores = []
            for i in range(len(gt_ch)):
                # inputs are expected to be RGB + have batch dimension
                curr_target = np.repeat(
                    gt_ch[i][None, ...], repeats=3, axis=0
                )
                curr_pred = np.repeat(
                    pred_ch[i][None, ...], repeats=3, axis=0
                )
                curr_target = _normalize_for_lpips([curr_target])
                curr_pred = _normalize_for_lpips([curr_pred])
                lpips_scores.append(
                    lpips(
                        prediction=curr_pred,
                        target=curr_target
                    )
                )
            lpips_list.append((np.mean(lpips_scores), compute_SE(lpips_scores)))

    return {
        "PSNR": psnr_list,
        "Pearson": pearson_list,
        "MicroSSIM": microssim_list,
        "MicroS3IM": ms3im_list,
        "SSIM": ssim_list,
        "MSSSIM": msssim_list,
        "LPIPS": lpips_list,
    }
    

def show_metrics(metrics_dict: dict[str]) -> None:
    print("---------------------------------------")
    for metric_name, values in metrics_dict.items():
        if not values:
            continue
        print(f"{metric_name}:")
        for i, (value, se) in enumerate(values):
            print(f"- Ch{i + 1}: {value:.3f} ± {se:.3f}")
            
            
def full_frame_evaluation(stitched_predictions, tar, inp, same_scale: bool = True):

    ncols = tar.shape[-1] + 1
    nrows = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    ax[0,0].imshow(inp)
    vmins = np.percentile(stitched_predictions, q=1, axis=[0, 1])
    vmaxes = np.percentile(stitched_predictions, q=99, axis=[0, 1])
    for i in range(ncols - 1):
        if same_scale:
            ax[0,i+1].imshow(tar[...,i], vmin=vmins[i], vmax=vmaxes[i])
            ax[1,i+1].imshow(stitched_predictions[...,i], vmin=vmins[i], vmax=vmaxes[i])
        else:
            ax[0,i+1].imshow(tar[...,i])
            ax[1,i+1].imshow(stitched_predictions[...,i])

        ax[0,i+1].set_title(f"Channel {i+1}", fontsize=15)

    # disable the axis for ax[1,0]
    ax[1,0].axis('off')
    ax[0,0].set_title("Input", fontsize=15)
    # set y labels on the right for ax[0,2]
    ax[0,ncols-1].yaxis.set_label_position("right")
    ax[0,ncols-1].set_ylabel("Target", fontsize=15)

    ax[1,ncols-1].yaxis.set_label_position("right")
    ax[1,ncols-1].set_ylabel("Predicted", fontsize=15)