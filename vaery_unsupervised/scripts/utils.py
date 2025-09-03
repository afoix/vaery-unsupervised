import os
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

LIGHTGRAY = "#d3d3d3"  # for zero/absent expression

def _expression_vector_for_gene(
    gene: str,
    cell_ids_in_order: Iterable,
    transcripts_df: pd.DataFrame,
    cell_col: str = "cell_id",
    gene_col: str = "gene",
    value_col: Optional[str] = "count",  # works if present; otherwise falls back to event counting
) -> np.ndarray:
    """
    Return expression aligned to `cell_ids_in_order`.
    Supports:
      - event-form: one row per (cell, gene) event (no `count` column)
      - long-form: rows with numeric `value_col` (e.g., 'count'), aggregated by sum
    Missing (cell,gene) -> 0.
    """
    # Subset rows for this gene
    sub = transcripts_df.loc[transcripts_df[gene_col] == gene]
    if sub.empty:
        return np.zeros(len(cell_ids_in_order), dtype=float)

    # Choose aggregation:
    use_event_counts = True
    if value_col is not None and value_col in sub.columns and np.issubdtype(sub[value_col].dtype, np.number):
        # numeric count column available -> sum it
        agg = sub.groupby(cell_col, as_index=True)[value_col].sum()
        use_event_counts = False
    if use_event_counts:
        # event-form: count rows per cell
        agg = sub.groupby(cell_col, as_index=True).size()

    # Map to provided order
    cell_ids_in_order = list(cell_ids_in_order)
    values = np.fromiter((agg.get(cid, 0.0) for cid in cell_ids_in_order), dtype=float, count=len(cell_ids_in_order))
    return values


def plot_gene_expression_panels(
    embeddings: Dict[str, np.ndarray],
    projections: Dict[str, np.ndarray],
    cell_ids_order: Iterable,
    marker_list: Iterable[str],
    transcripts_df: pd.DataFrame,
    *,
    save_dir: Optional[str] = None,
    file_prefix: str = "",
    show: bool = False,
    dpi: int = 300,
    point_size: float = 14.0,
    alpha: float = 0.9,
    cmap: str = "inferno",  # change to "Reds"/"Purples" if preferred
    include_colorbar: bool = False,
) -> Dict[str, Tuple[plt.Figure, np.ndarray]]:
    """
    Create static 2-row grids per gene:
      Row 1: embeddings (keys order)
      Row 2: projections (keys order)

    Coloring:
      - expression == 0 -> light gray
      - expression  > 0 -> single-hue gradient (cmap), normalized to positive range
    """
    names_emb = list(embeddings.keys())
    names_proj = list(projections.keys())
    if len(names_emb) == 0 and len(names_proj) == 0:
        raise ValueError("Provide at least one embedding or one projection.")

    # Validate shapes and N
    N = None
    for name, arr in list(embeddings.items()) + list(projections.items()):
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"'{name}' must be shape (N, 2); got {arr.shape}")
        if N is None:
            N = arr.shape[0]
        elif arr.shape[0] != N:
            raise ValueError("All embeddings/projections must have the same number of rows (cells).")

    cell_ids_order = list(cell_ids_order)
    if N is not None and len(cell_ids_order) != N:
        raise ValueError(f"cell_ids_order length ({len(cell_ids_order)}) != N ({N}).")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def _tidy_axis(ax, title):
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")

    out = {}
    n_cols = max(len(names_emb), len(names_proj))
    n_rows = 1 if (len(names_emb) == 0 or len(names_proj) == 0) else 2

    for gene in marker_list:
        expr = _expression_vector_for_gene(gene, cell_ids_order, transcripts_df)

        # Split zero vs positive; normalize only positives
        mask_pos = expr > 0
        expr = np.log1p(expr) # log normalize so cells with very highly expressed genes don't mess with dynamic range
        vmax = float(expr[mask_pos].max()) if mask_pos.any() else 1.0
        norm = Normalize(vmin=0.0, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=cmap)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]

        fig.suptitle(f"{gene} expression", fontsize=16, fontweight="bold")

        # Embeddings row
        for j, name in enumerate(names_emb):
            ax = axes[0, j]
            coords = np.asarray(embeddings[name])
            ax.scatter(coords[:, 0], coords[:, 1], s=point_size, c=LIGHTGRAY, alpha=alpha, linewidths=0)
            if mask_pos.any():
                ax.scatter(coords[mask_pos, 0], coords[mask_pos, 1],
                           s=point_size, c=expr[mask_pos], cmap=cmap, norm=norm, alpha=alpha, linewidths=0)
            _tidy_axis(ax, f"Embeddings {name}")

        # Projections row
        if n_rows == 2:
            for j, name in enumerate(names_proj):
                ax = axes[1, j]
                coords = np.asarray(projections[name])
                ax.scatter(coords[:, 0], coords[:, 1], s=point_size, c=LIGHTGRAY, alpha=alpha, linewidths=0)
                if mask_pos.any():
                    ax.scatter(coords[mask_pos, 0], coords[mask_pos, 1],
                               s=point_size, c=expr[mask_pos], cmap=cmap, norm=norm, alpha=alpha, linewidths=0)
                _tidy_axis(ax, f"Projections {name}")

        if include_colorbar and mask_pos.any():
            cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, aspect=30)
            cbar.set_label("Expression", rotation=270, labelpad=15)

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save_dir is not None:
            fname = f"{file_prefix}{gene}_expression.png" if file_prefix else f"{gene}_expression.png"
            fpath = os.path.join(save_dir, fname)
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        out[gene] = (fig, axes)

    return out
