#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import torch
import tifffile as tiff
from iohub.ngff import open_ome_zarr

# ---------------- CONFIG ----------------
PATH_TO_TIFFS = Path("/Users/afoix/Downloads/test_tiff/")   # <- your folder
OUTPUT_ZARR   = Path("/Users/afoix/Downloads/output_omezarr/dataset.zarr")
OVERWRITE     = True
# ----------------------------------------


def read_tiff_to_tczyx(path: Path) -> np.ndarray:
    """
    Read a TIFF and return a numpy array shaped (T, C, Z, Y, X).
    Robust to axes like YX, ZYX, CYX, CZYX, YXS (samples=channels), TYX, etc.
    """
    with tiff.TiffFile(str(path)) as tf:
        arr = tf.series[0].asarray()
        axes = tf.series[0].axes  # e.g. 'YX', 'CYX', 'YXS', 'TCZYX'
    # Treat 'S' (samples) as channels
    axes = axes.replace('S', 'C')

    data = np.asarray(arr)
    current_axes = list(axes)  # e.g. ['Y','X'] or ['C','Y','X']

    # Insert missing axes so we can safely transpose later
    target_axes = ['T', 'C', 'Z', 'Y', 'X']
    for ax in target_axes:
        if ax not in current_axes:
            data = np.expand_dims(data, axis=0)
            current_axes = [ax] + current_axes  # we added it in front

    # Now permute to T,C,Z,Y,X
    perm = [current_axes.index(ax) for ax in target_axes]
    data = np.transpose(data, axes=perm)

    # Ensure 5D
    assert data.ndim == 5, f"Failed to normalise to 5D for {path} (got {data.shape})"
    return data


def main():
    files = sorted([p for p in PATH_TO_TIFFS.iterdir() if p.suffix.lower() in {".tif", ".tiff"}])
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {PATH_TO_TIFFS}")

    # Read first file to determine channel count & spatial shape
    first = read_tiff_to_tczyx(files[0])
    T0, C, Z, Y, X = first.shape

    # Auto channel names
    channel_names = [f"channel_{i}" for i in range(C)]

    # Accumulate along T (time) axis
    tensors = [torch.from_numpy(first)]
    for p in files[1:]:
        arr = read_tiff_to_tczyx(p)
        # sanity check: same C,Z,Y,X
        if arr.shape[1:] != (C, Z, Y, X):
            raise ValueError(
                f"Shape mismatch for {p.name}: got {arr.shape}, expected (*,{C},{Z},{Y},{X})."
            )
        tensors.append(torch.from_numpy(arr))

    stack = torch.cat(tensors, dim=0)  # (T_total, C, Z, Y, X)

    # Write to OME-Zarr
    OUTPUT_ZARR.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if OVERWRITE else "w-"
    with open_ome_zarr(str(OUTPUT_ZARR), mode=mode, layout="fov", channel_names=channel_names) as store:
        store["img"] = stack.numpy()

    print(f"Wrote {stack.shape[0]} timepoints with C={C} to {OUTPUT_ZARR}")
    print(f"Final shape (T,C,Z,Y,X): {tuple(stack.shape)}")


if __name__ == "__main__":
    main()
