"""Pick out early-sampled tips whose lineages persist to a later sample."""

import typing

import numba as nb
import numpy as np
import pandas as pd
from phyloframe import legacy as pfl


@nb.njit(cache=True)
def _mark_persistent_tips(
    ancestor_ids: np.ndarray, is_early: np.ndarray, is_late: np.ndarray
) -> np.ndarray:
    """Implementation; requires contiguous ids in topological order."""
    n = len(ancestor_ids)
    has_early, has_late = is_early.copy(), is_late.copy()
    for i in range(n - 1, 0, -1):
        has_early[ancestor_ids[i]] |= has_early[i]
        has_late[ancestor_ids[i]] |= has_late[i]

    # a junction is where a branch holding only late tips joins a clade
    # that holds early tips
    is_junction = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        a = ancestor_ids[i]
        if a != i and has_late[i] and not has_early[i] and has_early[a]:
            is_junction[a] = True

    # nearest junction at or above each node, across late-free branches only
    home = np.full(n, -1, dtype=np.int64)
    # topmost node of the late-free subtree holding each late-free node
    top = np.full(n, -1, dtype=np.int64)
    # junction that each early-free, late-bearing node hangs off
    junction = np.full(n, -1, dtype=np.int64)
    # early tip marked persistent for each junction
    representative = np.full(n, -1, dtype=np.int64)
    is_claimed = np.zeros(n, dtype=np.bool_)
    is_persistent = np.zeros(n, dtype=np.bool_)
    is_transient = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        a = ancestor_ids[i]
        if is_junction[i]:
            home[i] = i
        elif a != i and not has_late[i]:
            home[i] = home[a]
        if not has_late[i]:
            top[i] = top[a] if a != i and not has_late[a] else i
        elif a != i and not has_early[i]:
            junction[i] = a if has_early[a] else junction[a]
        if not is_early[i]:
            continue
        if home[i] >= 0:
            if not is_claimed[home[i]]:
                is_claimed[home[i]] = True
                is_persistent[i] = True
                representative[home[i]] = i
        elif not is_claimed[top[i]]:
            is_claimed[top[i]] = True
            is_transient[i] = True

    # persistent early tip standing for the lineage of each late tip
    relative = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if is_late[i] and junction[i] >= 0:
            relative[i] = representative[junction[i]]
    return is_persistent, is_transient, home >= 0, relative


def mark_persistent_tips(
    phylogeny_df: pd.DataFrame,
    *,
    early: str,
    late: str,
    mark_as: str = "is_persistent",
    mark_transient_as: typing.Optional[str] = None,
    representatives: bool = True,
    mark_relative_as: typing.Optional[str] = None,
) -> pd.DataFrame:
    """Mark one early tip for each lineage that persists to a late tip.

    For every late tip, the nearest clade that also holds an early tip is
    found, by walking rootward. The node where the late tip's branch joins
    that clade is a "junction": the point where a lineage sampled at the
    early time point is last seen before going on to leave late descendants.
    One early tip is marked per junction, taken from among the early tips
    that reach the junction without passing any other branch with late
    descendants (i.e., the late tip's closest early relatives).

    Which of those equally close early tips is marked is arbitrary, so the
    marked tip need not itself be an ancestor of the late tips.

    Optionally, one early tip is also marked for each transient lineage: a
    clade of early tips without late tips that is not among the closest early
    relatives of any late tip.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Asexual phylogeny in alife standard format.
    early, late : str
        Names of boolean columns flagging early- and late-sampled tips.
    mark_as : str, default "is_persistent"
        Name of boolean output column.
    mark_transient_as : str, optional
        Name of boolean output column for transient lineages, if wanted.
    mark_relative_as : str, optional
        Name of output column giving, for each late tip, the id of the early
        tip marked persistent for its lineage (-1 elsewhere, or if none).
        Ids are those of the returned phylogeny.
    representatives : bool, default True
        If False, every early tip is marked instead of one per lineage:
        persistent if it is among the closest early relatives of some late
        tip, and transient otherwise.

    Returns
    -------
    pd.DataFrame
        Phylogeny, topologically sorted with contiguous ids, with the output
        column(s) added.
    """
    phylogeny_df = pfl.alifestd_try_add_ancestor_id_col(phylogeny_df)
    phylogeny_df = pfl.alifestd_topological_sort(phylogeny_df)
    phylogeny_df = pfl.alifestd_assign_contiguous_ids(phylogeny_df)
    phylogeny_df = phylogeny_df.reset_index(drop=True)
    is_early = phylogeny_df[early].to_numpy(dtype=np.bool_)
    is_persistent, is_transient, has_home, relative = _mark_persistent_tips(
        phylogeny_df["ancestor_id"].to_numpy(dtype=np.int64),
        is_early,
        phylogeny_df[late].to_numpy(dtype=np.bool_),
    )
    if not representatives:
        is_persistent = is_early & has_home
        is_transient = is_early & ~has_home
    phylogeny_df[mark_as] = is_persistent
    if mark_transient_as is not None:
        phylogeny_df[mark_transient_as] = is_transient
    if mark_relative_as is not None:
        phylogeny_df[mark_relative_as] = relative
    return phylogeny_df
