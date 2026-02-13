import types
import typing

import alifedata_phyloinformatics_convert as apc
import downstream
import hstrat
from hstrat._auxiliary_lib import alifestd_as_newick_asexual
from hstrat.dataframe import surface_unpack_reconstruct
from hstrat.phylogenetic_inference import (
    AssignOriginTimeNodeRankTriePostprocessor,
)
import iplotx as ipx
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from teeplot import teeplot as tp
import tqdm

np.bool = bool


class MultiTracker():

    _supported_S = {1, 2, 4, 8, 16, 32, 64}

    def _get_int_dtype(self):
        if self.bitwidth >= 64: return self.np.uint64
        if self.bitwidth >= 32: return self.np.uint32
        if self.bitwidth >= 16: return self.np.uint16
        return self.np.uint8

    def __init__(
        self,
        dstream_S: int,
        dstream_bitwidth: int,
        dstream_algo: types.ModuleType,
        *,
        N: int = 1,
        backend: types.ModuleType = np
    ):
        assert dstream_S in self._supported_S, f"dstream_S must be in {self._supported_S}"
        self.algo = dstream_algo
        self.S = dstream_S
        self.bitwidth = dstream_bitwidth
        self.N = N
        self.np = backend
        self.rng = self.np.random.default_rng()
        self.marker_dtype = self._get_int_dtype()

    def initialize(self, data: np.ndarray, R: int):
        if data.dtype != self.np.bool_:
            print("Warning: casting data to bool")
            data = data.astype(self.np.bool_)

        assert isinstance(data, self.np.ndarray), "data must match the backend"

        H, W = data.shape
        self.H = H 
        self.W = W
        self.R = R

        self.hst_markers = self.np.random.randint(
            0, 2**self.bitwidth, size=(self.N, H, W, self.S), dtype=self.marker_dtype
        )  # deposit random stuff everywhere first gen

        self.ancestor_search = [
            (i, j) for i in range(-R, R + 1) for j in range(-R, R + 1)
        ]
        parent_idx = self.np.arange(0, H * W, dtype=np.uint32).reshape(H, W)
        parent_keys = []
        for x, y in self.ancestor_search:
            arr = self.np.zeros((H, W), dtype=self.np.uint32)
            arr[*self._get_slices(-x, -y)] = parent_idx[*self._get_slices(x, y)]
            parent_keys.append(arr[None])
        self.parents = self.np.concatenate(parent_keys)

        self.curr = data
        self.t = 0
        self.history = []
        return self
    
    def step(self, data, *, save: bool = False, add_random_noise: bool = False):
        self.t += 1

        assert isinstance(data, self.np.ndarray), "data must match the backend"

        if data.dtype != self.np.bool_:
            print("Warning: casting data to bool")
            data = data.astype(self.np.bool_)

        parent = self.curr

        ancestor_matrices = []
        for x, y in self.ancestor_search:
            arr = self.np.zeros_like(data, dtype=self.np.float16)
            arr[*self._get_slices(-x, -y)] = parent[*self._get_slices(x, y)]
            ancestor_matrices.append(arr[None])
        scores = self.np.concatenate(ancestor_matrices)[None].repeat(self.N, axis=0)

        if add_random_noise:
            if self.N == 1 and self.t == 1:  # only print this once
                print("Warning: It does not make sense to add noise with N=1")
            scores += self.rng.uniform(low=0, high=0.5, size=scores.shape).astype(self.np.float16)  # 0.5 to ensure 0's don't beat 1's 

        winning_scores = self.parents.transpose(1, 2, 0)[None].repeat(self.N, axis=0)[
            self.np.eye(scores.shape[1], dtype=self.np.bool_)[scores.argmax(axis=1)]
        ].reshape(self.N, self.H, self.W)
        self.hst_markers[:] = self.hst_markers[
            self.np.arange(self.hst_markers.shape[0]).reshape(self.hst_markers.shape[0], 1, 1),
            winning_scores // self.W, 
            winning_scores % self.W
        ]

        item_to_assign = self.algo.assign_storage_site(self.S, self.t)
        if item_to_assign is not None:
            self.hst_markers[..., item_to_assign] = self.np.random.randint(
                0, 2**self.bitwidth, size=(self.N, self.H, self.W), dtype=self.marker_dtype
            )
        if save:
            self.history.append((self.t, data.copy(), self.hst_markers.copy())) 
        self.curr = data
        return self
    
    def to_numpy(self, x):
        if not isinstance(x, np.ndarray):
            return x.get()
        return x
    
    def reconstruct_phylogenies(self, *, verbose=False):
        self.history.append((self.t, self.curr.copy(), self.hst_markers.copy()))

        results = []
        for idx in range(self.N):
            extant_information = [
                (
                    t,
                    self.to_numpy(self.np.argwhere(ca_state)),
                    np.apply_along_axis(
                        self._pack_hex,
                        -1, self.to_numpy(hst_markers[idx, ca_state])
                    )
                )
                for t, ca_state, hst_markers in self.history
            ]

            population = []
            for t, positions, genomes in extant_information:
                assert len(positions) == len(genomes)
                for (y, x), i in zip(positions, genomes):
                    population.append(
                        {
                            "downstream_version": downstream.__version__,
                            "data_hex": np.asarray(t, dtype=">u4").tobytes().hex() + i,
                            "dstream_algo": f"dstream.{self.algo.__name__.split('.')[-1]}",
                            "dstream_storage_bitoffset": 32,
                            "dstream_storage_bitwidth": self.S * self.bitwidth,
                            "dstream_T_bitoffset": 0,
                            "dstream_T_bitwidth": 32,
                            "dstream_S": self.S,
                            "extant": t == self.t,
                            "row": y,
                            "col": x,
                            "state": 1
                        }
                    )

            postprocessor = AssignOriginTimeNodeRankTriePostprocessor(t0="dstream_S")
            df = surface_unpack_reconstruct(pl.from_pandas(pd.DataFrame(population))).to_pandas()
            df["rank"] = df["hstrat_rank"]
            df = postprocessor(df, 2**-1)
            results.append(df)
        return results

    def _get_slices(self, x, y):
        x_slice = slice(0, self.W + x) if x < 0 else slice(x, self.W)
        y_slice = slice(0, self.H + y) if y < 0 else slice(y, self.H)
        return y_slice, x_slice 
    
    def _pack_hex(self, items) -> str:
        if self.bitwidth == 1:
            packed_bytes = np.packbits(items, bitorder="big").tobytes()
        elif self.bitwidth == 4:
            packed_bytes = ((items[0::2] << 4) | items[1::2]).tobytes()
        elif self.bitwidth & 7 == 0:
            packed_bytes = items.astype(f">u{self.bitwidth >> 3}").tobytes()
        return packed_bytes.hex()



class CATracker(MultiTracker):

    def __init__(
        self,
        dstream_S: int,
        dstream_bitwidth: int,
        dstream_algo: types.ModuleType,
        *,
        backend: types.ModuleType = np
    ):
        super().__init__(dstream_S, dstream_bitwidth, dstream_algo, N=1, backend=backend)

    def reconstruct_phylogeny(self, *, verbose=False):
        df = self.reconstruct_phylogenies(verbose=verbose)[0]
        return df, alifestd_as_newick_asexual(df)


# todo think about adding noise to curr (to make random ones stronger) or the scores
def track_ca_history(
    data: np.ndarray,
    dstream_algo: types.ModuleType,
    dstream_S: int,
    dstream_bitwidth: int,
    R: int,
    *,
    fossil_range: typing.Container[int] = set(),
    add_random_noise: bool = False,
) -> typing.Tuple[pd.DataFrame, str]:
    """
    Arguments
    =========
    data: np.ndarray
        Boolean array in the format (timesteps, world height, world width)
    algo: types.ModuleType
        Downstream algorithm to determine where to deposit stratum
    S: int
        Surface size of underlying dstream surface
    W: int
        Radius of possible ancestors

    Returns
    =======
    An alifestd DataFrame containing genome information (for use with
    hstrat.dataframe.surface_unpack_reconstruct).
    """
    if isinstance(data, np.ndarray):
        backend = np 
    else:
        try: 
            import cupy as cp
            if isinstance(data, cp.ndarray):
                backend = cp
            else:
                raise TypeError("data must be either a numpy or cupy ndarray")
        except ImportError:
            pass

    ca_tracker = CATracker(
        dstream_S=dstream_S, 
        dstream_algo=dstream_algo, 
        dstream_bitwidth=dstream_bitwidth,
        backend=backend
    )
    ca_tracker.initialize(data[0], R=R)
    for i, curr in tqdm.tqdm(enumerate(data[1:], start=1), total=data.shape[0]-1):
        ca_tracker.step(curr, save=i in fossil_range, add_random_noise=add_random_noise)
    return ca_tracker.reconstruct_phylogeny()


def plot_phylo_at(phylo_df: pd.DataFrame, teeplot_subdir: str, show: bool = True, fossil_rank: typing.Optional[int] = None) -> None:
    if fossil_rank is None:
        fossil_rank = phylo_df["hstrat_rank"].max() + 1
    df = phylo_df[phylo_df["hstrat_rank"] < fossil_rank].copy()
    df.loc[(df["hstrat_rank"] == fossil_rank - 1) & (df["state"] == 1), "extant"] = True
    del df["origin_time"]
    dp_tree = apc.alife_dataframe_to_dendropy_trees(
        hstrat._auxiliary_lib.alifestd_try_add_ancestor_list_col(df),
        setup_edge_lengths=True,
    )
    grid_dim = int(df[df["gol_state"] >= 0][["row", "col"]].to_numpy().max()) + 1

    with tp.teed(
        plt.subplots,
        1,
        2,
        figsize=(10, 10),
        gridspec_kw={
            'width_ratios': [0.2, 0.8],
            'height_ratios': [1],
            'wspace': 0.05,
            'hspace': 0.05,
        },
        teeplot_subdir=teeplot_subdir,
        teeplot_outattrs={"rank": fossil_rank},
        teeplot_show=show
        
    ) as teed:
        fig, (ax_left, ax_grid) = teed

        grid = np.full((grid_dim, grid_dim), np.nan)
        for _, row in df[(df["gol_state"] >= 0) & df["extant"]].iterrows():
            grid[int(row["row"]), int(row["col"])] = 0
            
        cmap = sns.color_palette("Dark2", 1)

        tree_left = ipx.plotting.tree(
            dp_tree[0],
            ax=ax_left,
            edge_color=to_hex(cmap[0]),
            edge_linewidth=0.5,
            layout_angular=True,
            margins=0.0,
        )
        ax_left.invert_yaxis()
        ax_left.margins(y=-0.05)

        sns.heatmap(
            grid,
            ax=ax_grid,
            cmap=cmap,
            vmin=0,
            vmax=len(dp_tree) - 1,
            cbar=False,
        )
        ax_grid.set_axis_off()

        df = hstrat._auxiliary_lib.alifestd_mark_leaves(df, mutate=True)

        for idx, row in df[df["is_leaf"] & df["extant"]].iterrows():

            tree_x, tree_y = next(
                v for n, v in tree_left.get_layout().T.items()
                if n.taxon is not None and n.taxon.label == row["id"]
            )
            grid_x, grid_y = row["col"], row["row"]

            # draw line between axes from (tree_x, tree_y) to (grid_x, grid_y)
            con = ConnectionPatch(
                xyA=(tree_x, tree_y),
                xyB=(grid_x, grid_y),
                coordsA=ax_left.transData,
                coordsB=ax_grid.transData,
                color="gray",
                linestyle="--",
                alpha=0.5,
                linewidth=0.5,
                clip_on=False,
            )
            fig.add_artist(con)


def plot_ancestry_at(phylo_df: pd.DataFrame, parent_rank: int, teeplot_subdir: str, child_rank: typing.Optional[int] = None, show: bool = False) -> None:
    if child_rank is None:
        child_rank = phylo_df["hstrat_rank"].max() + 1
    df_by_id = phylo_df.set_index("id", inplace=False)
    parent_taxa, child_taxa = (
        set(phylo_df.loc[(phylo_df["hstrat_rank"] == rank - 1) & (phylo_df["state"] == 1), "taxon_label"])
        for rank in (parent_rank, child_rank)
    )
    dp_tree = apc.alife_dataframe_to_dendropy_trees(
        hstrat._auxiliary_lib.alifestd_try_add_ancestor_list_col(phylo_df),
        setup_edge_lengths=True,
    )

    df_no_origin_time = phylo_df.drop(columns=["origin_time"])
    dp_tree_for_plotting = apc.alife_dataframe_to_dendropy_trees(
        hstrat._auxiliary_lib.alifestd_try_add_ancestor_list_col(df_no_origin_time),
        setup_edge_lengths=True,
    )
    pdm = dp_tree[0].phylogenetic_distance_matrix()

    # determine the most recent parent for each child
    ancestors = [
        max((
            (parent_id, df_by_id.at[pdm.mrca(child_id, parent_id).taxon.label, "hstrat_rank"]) 
            for parent_id in dp_tree[0].taxon_namespace if parent_id.label in parent_taxa
        ), key=lambda x: x[1])[0].label
        for child_id in dp_tree[0].taxon_namespace if child_id.label in child_taxa
    ]

    with tp.teed(
        plt.subplots,
        2,
        2,
        figsize=(20, 12),
        gridspec_kw={
            'width_ratios': [0.5, 0.5],
            'height_ratios': [0.2, 0.8],
            'wspace': 0.05,
            'hspace': 0.05,
        },
        teeplot_subdir=teeplot_subdir,
        teeplot_outattrs={
            "parent-rank": parent_rank,
            "child-rank": child_rank
        },
        teeplot_show=show
    ) as teed:
        
        fig, ((ax_top1, ax_top2), (ax_left, ax_right)) = teed
        gs = ax_top1.get_gridspec()
        ax_top1.remove()
        ax_top2.remove()
        ax_top = fig.add_subplot(gs[0, :])
        tree = ipx.plotting.tree(
            dp_tree_for_plotting[0],
            ax=ax_top,
            edge_color="black",
            edge_linewidth=0.5,
            layout_angular=True,
            layout="vertical",
            margins=0.0,
        )
        ax_top.margins(x=-0.04)
        ax_top.set_xlim(ax_top.get_xlim()[0] - 10, None)
 

        for i, (rank, ax) in enumerate([(parent_rank, ax_left), (child_rank, ax_right)]):

            # create grid of the CA cells
            grid_dim = int(phylo_df[(phylo_df["hstrat_rank"] == rank - 1) & (phylo_df["gol_state"] >= 0)][["row", "col"]].to_numpy().max()) + 1
            grid = np.full((grid_dim, grid_dim), np.nan)
            for _, row in phylo_df[(phylo_df["hstrat_rank"] == rank - 1) & (phylo_df["gol_state"] >= 0)].iterrows():
                grid[int(row["row"]), int(row["col"])] = 1

            # plot heatmap
            sns.heatmap(
                grid,
                ax=ax,
                vmin=1,
                vmax=1,
                cmap=ListedColormap(sns.color_palette(["#0099ff" if i == 1 else "#ff0000"])),
                cbar=False,
            )
            # ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.patch.set_edgecolor('black')
            ax.patch.set_linewidth(1)

            for _, row in phylo_df[(phylo_df["hstrat_rank"] == rank - 1) & (phylo_df["gol_state"] >= 0)].iterrows():

                tree_x, tree_y = next(
                    v for n, v in tree.get_layout().T.items()
                    if n.taxon is not None and n.taxon.label == row["id"]
                )
                grid_x, grid_y = row["col"], row["row"]

                # draw line between axes from (tree_x, tree_y) to (grid_x, grid_y)
                con = ConnectionPatch(
                    xyA=(tree_x, tree_y),
                    xyB=(grid_x, grid_y),
                    coordsA=ax_top.transData,
                    coordsB=ax.transData,
                    color="#ff0000" if i == 0 else "#0099ff",
                    linestyle="--",
                    alpha=0.2,
                    linewidth=0.5,
                    clip_on=False,
                )
                fig.add_artist(con)

        for (child, parent) in zip(child_taxa, ancestors):
            child_y, child_x = df_by_id.at[child, "row"], df_by_id.at[child, "col"]
            parent_y, parent_x = df_by_id.at[parent, "row"], df_by_id.at[parent, "col"]
            con = ConnectionPatch(
                xyA=(parent_x, parent_y),
                xyB=(child_x, child_y),
                coordsA=ax_left.transData,
                coordsB=ax_right.transData,
                color="gray",
                linestyle="--",
                alpha=0.2,
                linewidth=0.5,
                clip_on=False,
            )
            fig.add_artist(con)
