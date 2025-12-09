import types
import typing

import seaborn as sns
from teeplot import teeplot as tp
import dendropy as dp
import downstream
import hstrat
from hstrat.phylogenetic_inference import AssignOriginTimeNodeRankTriePostprocessor
from hstrat.dataframe import surface_build_tree, surface_unpack_reconstruct
from hstrat._auxiliary_lib import alifestd_as_newick_asexual
import numpy as np
import pandas as pd
import polars as pl
import tqdm
from matplotlib.colors import to_hex, ListedColormap
import iplotx as ipx
import alifedata_phyloinformatics_convert as apc
from matplotlib.patches import ConnectionPatch
from matplotlib import pyplot as plt

np.bool = bool    

class CATracker:

    def __init__(
        self,
        H: int, W: int, R: int,
        dstream_S: int,
        dstream_algo: types.ModuleType,
    ):
        
        assert dstream_S in {
            8,
            16,
            32,
            64,
        }, "Surface size must be a power of two between 8 and 64"
            
        self.algo = dstream_algo
        self.S = dstream_S
        self.R = R
        self.H = H 
        self.W = W

        self.hst_markers = np.random.randint(
            0, 2, size=(H, W, dstream_S), dtype=np.bool_
        )  # deposit random stuff everywhere first gen

        self.ancestor_search = [
            (i, j) for i in range(-R, R + 1) for j in range(-R, R + 1)
        ]
        parent_idx = np.arange(0, H * W).reshape(H, W)
        parent_keys = []
        for x, y in self.ancestor_search:
            arr = np.zeros((H, W), dtype=np.int64)
            arr[*self._get_slices(-x, -y)] = parent_idx[*self._get_slices(x, y)]
            parent_keys.append(arr[None])
        self.parents = np.concatenate(parent_keys)

    def initialize(self, data: np.ndarray):
        self.curr = data
        self.t = 0
        self.history = []
        return self

    def step(self, data: np.ndarray, *, save: bool = False, add_random_noise: bool = False):
        self.t += 1

        bit_to_assign = self.algo.assign_storage_site(self.S, self.t)
        parent = self.curr

        ancestor_matrices = []
        for x, y in self.ancestor_search:
            arr = np.zeros_like(data, dtype=np.int64)
            arr[*self._get_slices(-x, -y)] = parent[*self._get_slices(x, y)]
            ancestor_matrices.append(arr[None])
        scores = np.concatenate(ancestor_matrices).astype(np.float32)
        if add_random_noise:
            scores += np.random.uniform(low=0, high=0.5, size=scores.shape)  # 0.5 to ensure 0's don't beat 1's 

        winning_scores = self.parents.transpose(1, 2, 0)[
            np.eye(scores.shape[0], dtype=np.bool_)[scores.argmax(axis=0)]
        ].reshape(self.H, self.W)
        self.hst_markers[:] = self.hst_markers[winning_scores // self.W, winning_scores % self.W]
        if bit_to_assign is not None:
            self.hst_markers[:, :, bit_to_assign] = np.random.randint(
                0, 2, size=(self.H, self.W), dtype=np.bool_
            )
        if save:
            self.history.append((self.t, data.copy(), self.hst_markers.copy())) 
        self.curr = data
        return self


    def reconstruct_phylogeny(self, *, verbose=False):
        self.history.append((self.t, self.curr.copy(), self.hst_markers.copy()))

        if verbose:
            print("Generating extant information...")

        extant_information = [
            (t, np.argwhere(ca_state), np.packbits(hst_markers[ca_state], axis=-1, bitorder="big").view(dtype=f">u{self.S // 8}")) 
            for t, ca_state, hst_markers in self.history
        ]

        if verbose:
            print("Creating population...")

        population = []
        for t, positions, genomes in extant_information:
            assert len(positions) == len(genomes)
            for (y, x), i in zip(positions, genomes):
                population.append(
                    {
                        "downstream_version": downstream.__version__,
                        "data_hex": np.asarray(t, dtype=np.uint32)
                        .astype(">u4")
                        .tobytes()
                        .hex()
                        + i.tobytes().hex(),
                        "dstream_algo": f"dstream.{self.algo.__name__.split('.')[-1]}",
                        "dstream_storage_bitoffset": 32,
                        "dstream_storage_bitwidth": self.S,
                        "dstream_T_bitoffset": 0,
                        "dstream_T_bitwidth": 32,
                        "dstream_S": self.S,
                        "extant": t == self.t,
                        "row": y,
                        "col": x,
                        "state": 1
                    }
                )

        if verbose:
            print("Reconstructing phylogeny from CA history...")

        postprocessor = AssignOriginTimeNodeRankTriePostprocessor(t0="dstream_S")
        df = surface_unpack_reconstruct(pl.from_pandas(pd.DataFrame(population))).to_pandas()
        df["rank"] = df["hstrat_rank"]
        df = postprocessor(df, 2**-1)
        return df, alifestd_as_newick_asexual(df)


    def _get_slices(self, x, y):
        x_slice = slice(0, self.W + x) if x < 0 else slice(x, self.W)
        y_slice = slice(0, self.H + y) if y < 0 else slice(y, self.H)
        return y_slice, x_slice 



# todo think about adding noise to curr (to make random ones stronger) or the scores
def track_ca_history(
    data: np.ndarray,
    algo: types.ModuleType,
    S: int,
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
    H, W = data.shape[1:]
    ca_tracker = CATracker(
        H, W, R,
        dstream_S=S,
        dstream_algo=algo,
    )
    ca_tracker.initialize(data[0])
    for i, curr in tqdm.tqdm(enumerate(data[1:], start=1)):
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