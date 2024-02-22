import pandas as pd
import polars as pl

from ..utils import to_list


class Indexer:
    def __init__(self, indexer):
        if not isinstance(indexer, pd.DataFrame):
            raise TypeError
        if not {"locs", "n_rows"} <= set(indexer.columns):
            raise ValueError
        self.indexer = indexer
        self.indices = indexer.index.to_numpy()

    @property
    def index_cols(self):
        index_cols = tuple(self.indexer.index.names)
        if len(index_cols) == 1:
            return index_cols[0]
        return index_cols

    @classmethod
    def from_dataframe(cls, df, index_cols):
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df, rechunk=False, include_index=False)
        elif not isinstance(df, pl.DataFrame):
            raise TypeError

        index_cols = to_list(index_cols)
        indexer = df.with_row_index("locs").group_by(index_cols).agg(pl.col("locs"), pl.count("locs").alias("n_rows"))
        indexer = indexer.sort(by=index_cols).to_pandas(use_pyarrow_extension_array=True)
        indexer.set_index(index_cols, inplace=True)
        return cls(indexer)

    def get_locs(self, indices, return_n_rows=False):
        subset = self.indexer.loc[indices]
        locs = subset["locs"].list.flatten().to_numpy()

        if return_n_rows:
            n_rows = subset["n_rows"].to_numpy()
            return locs, n_rows
        return locs
