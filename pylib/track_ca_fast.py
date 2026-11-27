"""Compiled, sparse-aware drop-in for `track_ca.MultiTracker` on large runs.

`track_ca.MultiTracker` touches every site of the grid nine times per step
(once per candidate parent offset), regardless of whether the site is alive.
That is fine on a GPU at 2,300 x 2,300 sites, but does not scale to much
larger runs on CPU.

`FastCATracker` fuses the cellular automaton update and the hstrat marker
inheritance into a single compiled pass, with the same semantics as running

    grid = update(grid, rule_table)
    tracker.step(grid, add_random_noise=True)

with `MultiTracker(dstream_S=64, dstream_bitwidth=1, N=1)`. Specifically,

1. the automaton is advanced with the same 512-entry Moore neighborhood
   lookup table and neighborhood bit packing as `pack_neighborhoods`,
2. each live cell inherits the markers of one parent, drawn uniformly at
   random from the cells of its Moore neighborhood (self included) that were
   alive at the previous step, and
3. every cell then deposits one fresh random differentia bit at the surface
   site picked by `dstream_algo.assign_storage_site(S, t)`, if any.

The speedup comes from four places, none of which alter those semantics.

- Only live cells are tracked. With `rule[0] == 0`, every live cell has at
  least one live candidate parent, and live candidates always outscore dead
  ones, so markers held by dead cells can never be inherited by a live cell
  and are never observed.
- The 64 single-bit differentia of a cell are packed into one `uint64`, so
  inheritance is one word copy and a deposit is one bit write.
- Empty space is skipped eight sites at a time, and only the bounding box of
  the pattern is visited.
- The pass is compiled with numba and parallelized over rows.

The deliberate deviations are as follows.

- Randomness comes from a counter-based hash keyed on `(seed, t, row, col)`,
  so runs are reproducible.
- Parent choice is exactly uniform (`MultiTracker` breaks rare float16 ties
  in favor of the first candidate offset).
- Genomes report `t + 1` deposits rather than `t`, counting the initial fill
  of site 0 along with the deposits at steps 1 through `t`. Reporting `t`
  leads reconstruction to attribute the newest deposit to whichever older
  stratum previously held its site. That seldom matters under steady
  retention, which rarely deposits late in a run, but would corrupt a stratum
  under retention policies that deposit at every step.

Optionally, deposits can be made only every `time_dilation` steps, with the
downstream algorithm advanced once per deposit rather than once per step.
Marker inheritance still happens at every step. This spends the surface's
fixed number of sites on a coarser clock, which stretches the reach of
recency-weighted retention policies; reconstructed ranks are then in units of
`time_dilation` steps.
"""

import base64
import binascii
import types
import typing

import downstream
from hstrat.dataframe import surface_unpack_reconstruct
from hstrat.phylogenetic_inference import (
    AssignOriginTimeNodeRankTriePostprocessor,
)
import numba as nb
import numpy as np
import pandas as pd
import polars as pl

_S = 64  # surface size; one differentia bit per site, packed into a uint64
_NO_SITE = -1
_BIG = 1 << 62


def decode_rule_table(b64: str) -> np.ndarray:
    """Decode a base64-packed 512-entry Moore neighborhood rule table."""
    padded_bytes = base64.b64decode(b64 + "=" * (-len(b64) % 4))
    bits = np.unpackbits(np.frombuffer(padded_bytes, dtype=np.uint8))
    assert len(bits) == 512
    return bits.astype(np.uint8)


def make_lifelike_rule_table(
    birth: typing.Iterable[int], survive: typing.Iterable[int]
) -> np.ndarray:
    """Build a 512-entry rule table for an outer-totalistic (B/S) rule."""
    birth, survive = set(birth), set(survive)
    codes = np.arange(512)
    center = (codes >> 4) & 1
    neighbors = sum((codes >> i) & 1 for i in range(9)) - center
    alive = np.where(
        center == 1,
        np.isin(neighbors, sorted(survive)),
        np.isin(neighbors, sorted(birth)),
    )
    return alive.astype(np.uint8)


def parse_rle(rle_string: str) -> np.ndarray:
    """Parse the body of a run length encoded pattern."""
    rows, current_row, count = [], [], ""
    for char in rle_string:
        if char.isdigit():
            count += char
        elif char in "bo":
            current_row.extend([int(char == "o")] * (int(count or 1)))
            count = ""
        elif char == "$":
            rows.append(current_row)
            current_row, count = [], ""
        elif char == "!":
            break
    if current_row:
        rows.append(current_row)
    width = max(map(len, rows))
    return np.array(
        [row + [0] * (width - len(row)) for row in rows], dtype=np.uint8
    )


@nb.njit(inline="always")
def _mix(x):
    # splitmix64 finalizer
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@nb.njit(parallel=True, cache=True)
def _seed_markers(cells, markers, seed):
    """Deposit 64 random differentia bits at every live cell."""
    H, W = cells.shape
    for r in nb.prange(H):
        for x in range(W):
            if cells[r, x]:
                markers[r, x] = _mix(
                    _mix(seed + np.uint64(0x9E3779B97F4A7C15))
                    ^ np.uint64(r * W + x)
                )


@nb.njit(parallel=True, cache=True)
def _step(
    prev,  # uint8 (H, W), W % 8 == 0, state at t - 1
    new,  # uint8 (H, W), written with state at t
    prev8,  # uint64 (H, W // 8) view of prev
    new8,  # uint64 (H, W // 8) view of new
    mprev,  # uint64 (H, W), markers at t - 1
    mnew,  # uint64 (H, W), written with markers at t
    rule,  # uint8 (512,)
    r0,  # first row to update
    r1,  # last row to update, inclusive
    k0,  # first 8-site word to update
    k1,  # last 8-site word to update, inclusive
    site,  # surface site to deposit at, or _NO_SITE
    key,  # uint64 per-step random key
    track,  # whether to maintain markers
):
    W = prev.shape[1]
    nrows = r1 - r0 + 1
    row_lo = np.full(nrows, _BIG, np.int64)
    row_hi = np.full(nrows, -1, np.int64)
    row_n = np.zeros(nrows, np.int64)
    if site >= 0:
        site_shift = np.uint64(63 - site)  # packbits is big-endian
        keep = ~(np.uint64(1) << site_shift)
    else:
        site_shift = np.uint64(0)
        keep = ~np.uint64(0)

    for i in nb.prange(nrows):
        r = r0 + i
        a, b, c = prev[r - 1], prev[r], prev[r + 1]
        a8, b8, c8 = prev8[r - 1], prev8[r], prev8[r + 1]
        lo, hi, n = _BIG, -1, 0
        left = a8[k0 - 1] | b8[k0 - 1] | c8[k0 - 1]
        here = a8[k0] | b8[k0] | c8[k0]
        for k in range(k0, k1 + 1):
            right = a8[k + 1] | b8[k + 1] | c8[k + 1]
            if (left | here | right) == 0:
                new8[r, k] = 0
            else:
                for x in range(8 * k, 8 * k + 8):
                    code = (
                        np.int64(a[x - 1])
                        | (np.int64(a[x]) << 1)
                        | (np.int64(a[x + 1]) << 2)
                        | (np.int64(b[x - 1]) << 3)
                        | (np.int64(b[x]) << 4)
                        | (np.int64(b[x + 1]) << 5)
                        | (np.int64(c[x - 1]) << 6)
                        | (np.int64(c[x]) << 7)
                        | (np.int64(c[x + 1]) << 8)
                    )
                    alive = rule[code]
                    new[r, x] = alive
                    if not alive:
                        continue
                    if x < lo:
                        lo = x
                    hi = x
                    n += 1
                    if not track:
                        continue
                    # count candidate parents
                    count = 0
                    bits = code
                    while bits:
                        bits &= bits - 1
                        count += 1
                    h = _mix(key ^ np.uint64(r * W + x))
                    # uniform choice among live candidate parents
                    pick = np.int64(
                        ((h & np.uint64(0xFFFFFFFF)) * np.uint64(count))
                        >> np.uint64(32)
                    )
                    bits = code
                    for __ in range(pick):
                        bits &= bits - 1
                    j = 0
                    while not (bits >> j) & 1:
                        j += 1
                    m = mprev[r + j // 3 - 1, x + j % 3 - 1]
                    if site >= 0:
                        m = (m & keep) | ((h >> np.uint64(63)) << site_shift)
                    mnew[r, x] = m
            left, here = here, right
        row_lo[i], row_hi[i], row_n[i] = lo, hi, n

    R0, R1, C0, C1, N = _BIG, -1, _BIG, -1, 0
    for i in range(nrows):
        if row_n[i]:
            if R1 < 0:
                R0 = r0 + i
            R1 = r0 + i
            C0 = min(C0, row_lo[i])
            C1 = max(C1, row_hi[i])
            N += row_n[i]
    return R0, R1, C0, C1, N


class FastCATracker:
    """Run a Moore neighborhood CA with hstrat markers attached to cells.

    Equivalent to driving `track_ca.MultiTracker(dstream_S=64,
    dstream_bitwidth=1, N=1)` with `add_random_noise=True`; see module
    docstring.

    Parameters
    ----------
    rule : np.ndarray
        512-entry lookup table, indexed by the packed Moore neighborhood
        with bit `3 * (dy + 1) + (dx + 1)` holding the state of the neighbor
        at offset `(dy, dx)`.
    grid : np.ndarray
        Initial state, shape (H, W). The pattern must stay clear of the grid
        edges throughout the run (this is asserted), so boundary conditions
        never come into play.
    dstream_algo : module
        Downstream algorithm used to pick the site of each deposit.
    seed : int
        Seed for parent choice and differentia.
    time_dilation : int, default 1
        Number of steps per deposit.
    track : bool, default True
        If False, only the automaton is advanced.
    """

    def __init__(
        self,
        rule: np.ndarray,
        grid: np.ndarray,
        dstream_algo: types.ModuleType,
        *,
        seed: int = 1,
        time_dilation: int = 1,
        track: bool = True,
    ) -> None:
        rule = np.ascontiguousarray(rule, dtype=np.uint8)
        assert rule.shape == (512,) and rule.max() <= 1
        assert rule[0] == 0, "live cells must have a live candidate parent"
        self.rule = rule
        self.algo = dstream_algo
        self.S = _S
        self.bitwidth = 1
        self.seed = seed
        assert time_dilation >= 1
        self.time_dilation = time_dilation
        self.track = track
        self.t = 0

        H, W = grid.shape
        assert W % 8 == 0, "grid width must be a multiple of 8"
        self.H, self.W = H, W
        self._cells = [
            np.ascontiguousarray(grid, dtype=np.uint8).copy(),
            np.zeros((H, W), dtype=np.uint8),
        ]
        assert self._cells[0].max() <= 1
        self._markers = [
            np.zeros((H, W) if track else (1, 1), dtype=np.uint64)
            for __ in range(2)
        ]
        if track:
            _seed_markers(self._cells[0], self._markers[0], np.uint64(seed))

        rows, cols = np.nonzero(self._cells[0])
        assert len(rows), "initial state must not be empty"
        self.bbox = (rows.min(), rows.max(), cols.min(), cols.max())
        self.num_alive = len(rows)
        # region of each buffer that may hold stale nonzero state
        self._dirty = [self._pad(self.bbox), None]

    @staticmethod
    def _pad(bbox: tuple) -> tuple:
        r0, r1, c0, c1 = bbox
        return int(r0) - 1, int(r1) + 1, (int(c0) - 1) // 8, (int(c1) + 1) // 8

    @property
    def cells(self) -> np.ndarray:
        """Current state of the automaton."""
        return self._cells[self.t % 2]

    @property
    def markers(self) -> np.ndarray:
        """Packed differentia of live cells; bit `63 - s` is site `s`."""
        return self._markers[self.t % 2]

    @property
    def num_deposits(self) -> int:
        """Deposits made per cell, counting the initial fill of site 0."""
        return self.t // self.time_dilation + 1

    def step(self, num_steps: int = 1) -> "FastCATracker":
        for __ in range(num_steps):
            src, dst = self.t % 2, (self.t + 1) % 2
            self.t += 1
            r0, r1, k0, k1 = self._pad(self.bbox)
            if self._dirty[dst] is not None:  # overwrite all stale state
                q0, q1, p0, p1 = self._dirty[dst]
                r0, r1, k0, k1 = (
                    min(r0, q0),
                    max(r1, q1),
                    min(k0, p0),
                    max(k1, p1),
                )
            assert (
                r0 >= 1
                and k0 >= 1
                and r1 <= self.H - 2
                and k1 <= self.W // 8 - 2
            ), f"pattern reached the edge of the grid at step {self.t}"
            self._dirty[dst] = (r0, r1, k0, k1)

            site = (
                self.algo.assign_storage_site(
                    self.S, self.t // self.time_dilation
                )
                if self.t % self.time_dilation == 0
                else None
            )
            key = np.uint64(
                _mix(np.uint64(_mix(np.uint64(self.seed))) ^ np.uint64(self.t))
            )
            *bbox, self.num_alive = _step(
                self._cells[src],
                self._cells[dst],
                self._cells[src].view(np.uint64),
                self._cells[dst].view(np.uint64),
                self._markers[src],
                self._markers[dst],
                self.rule,
                r0,
                r1,
                k0,
                k1,
                _NO_SITE if site is None else int(site),
                key,
                self.track,
            )
            assert self.num_alive, f"pattern died out at step {self.t}"
            self.bbox = tuple(bbox)
        return self

    def make_population_df(self) -> pd.DataFrame:
        """Collate extant genomes, as by `MultiTracker`."""
        assert self.track
        cells = self.cells
        rows, cols = np.nonzero(cells)  # row-major, like np.argwhere
        genomes = np.empty((len(rows), 12), dtype=np.uint8)
        genomes[:, :4] = np.frombuffer(
            np.asarray(self.num_deposits, dtype=">u4").tobytes(),
            dtype=np.uint8,
        )
        genomes[:, 4:] = (
            self.markers[rows, cols]
            .astype(">u8")
            .view(np.uint8)
            .reshape(-1, 8)
        )
        data_hex = np.frombuffer(
            binascii.hexlify(genomes.tobytes()), dtype="S24"
        ).astype(str)
        return pd.DataFrame(
            {
                "downstream_version": downstream.__version__,
                "data_hex": data_hex,
                "dstream_algo": f"dstream.{self.algo.__name__.split('.')[-1]}",
                "dstream_storage_bitoffset": 32,
                "dstream_storage_bitwidth": self.S * self.bitwidth,
                "dstream_T_bitoffset": 0,
                "dstream_T_bitwidth": 32,
                "dstream_S": self.S,
                "extant": True,
                "row": rows,
                "col": cols,
                "state": 1,
            }
        )

    def reconstruct_phylogeny(self) -> pd.DataFrame:
        """Reconstruct phylogeny of extant cells, as by `MultiTracker`."""
        df = surface_unpack_reconstruct(
            pl.from_pandas(self.make_population_df())
        ).to_pandas()
        # column was renamed between hstrat releases
        rank = "hstrat_rank" if "hstrat_rank" in df else "dstream_rank"
        df["rank"] = df["hstrat_rank"] = df[rank]
        # inner nodes carry no tip metadata
        df["extant"] = df["extant"].eq(True)
        postprocessor = AssignOriginTimeNodeRankTriePostprocessor(
            t0="dstream_S"
        )
        return postprocessor(df, 2**-1)
