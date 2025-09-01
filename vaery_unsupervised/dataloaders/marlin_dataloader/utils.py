import pandas as pd


def expand_metadata_on_time(
    metadata: pd.DataFrame,
    n_timepoints: int,
    cols_to_keep: list[str] = None
) -> pd.DataFrame:
    metadata_expanded = (metadata
        .loc[:, cols_to_keep]
        .reset_index()
        .rename(columns={'index': 'gene_grna_trench_index'})
        .apply(lambda x: x.repeat(n_timepoints).reset_index(drop=True))
        .assign(timepoints=lambda df_: df_.index % n_timepoints,
                is_last_timepoint=lambda df_: df_['timepoints'] == n_timepoints - 1
        )
        .astype(
            {'Experiment #': 'category'}
        )
    )
    return metadata_expanded
