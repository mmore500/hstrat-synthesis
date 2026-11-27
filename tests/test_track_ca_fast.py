import itertools as it

from downstream import dstream
import numpy as np
import pytest
from scipy import stats as scipy_stats

from pylib.track_ca_fast import (
    FastCATracker,
    decode_rule_table,
    make_lifelike_rule_table,
    parse_rle,
)

OUTLIER_RULE = decode_rule_table(
    "ERETQB4eHWkQ7xD4eYZosBQZFixOBHmtFeehExrKVhURLRAqGxeIlSO1JYZP6DRi69rop7TQCkvWTIag7kAS8g",
)
OUTLIER_SEED = parse_rle("$bo$3o$2bo!")
SOUP = (np.random.default_rng(1).random((24, 24)) < 0.5).astype(np.uint8)
RULES = {
    "outlier": (OUTLIER_RULE, OUTLIER_SEED),
    "34life": (make_lifelike_rule_table([3, 4], [3, 4]), SOUP),
}


def make_grid(seed_pattern: np.ndarray, grid_size: int = 320) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    h, w = seed_pattern.shape
    oy, ox = (grid_size - h) // 2, (grid_size - w) // 2
    grid[oy : oy + h, ox : ox + w] = seed_pattern
    return grid


def reference_update(cells: np.ndarray, rule: np.ndarray) -> np.ndarray:
    """CA update as in pylib/track_outlier.ipynb."""
    cells = cells.astype(np.uint16)
    cells[:1], cells[-1:], cells[:, :1], cells[:, -1:] = 0, 0, 0, 0
    packed = np.zeros_like(cells)
    for i, (dy, dx) in enumerate(it.product((-1, 0, 1), repeat=2)):
        packed |= np.roll(cells << i, shift=(-dy, -dx), axis=(0, 1))
    return rule[packed]


def test_parse_rle():
    assert OUTLIER_SEED.tolist() == [
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 1],
    ]


def test_make_lifelike_rule_table_blinker():
    rule = make_lifelike_rule_table([3], [2, 3])
    grid = make_grid(np.array([[1, 1, 1]], dtype=np.uint8), 32)
    tracker = FastCATracker(rule, grid, dstream.steady_algo, track=False)
    assert tracker.step().cells[14:17, 14:17].tolist() == [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
    assert np.array_equal(tracker.step().cells, grid)


@pytest.mark.parametrize("name", RULES)
@pytest.mark.parametrize("track", [False, True])
def test_ca_matches_reference(name: str, track: bool):
    rule, seed_pattern = RULES[name]
    expected = make_grid(seed_pattern)
    tracker = FastCATracker(rule, expected, dstream.steady_algo, track=track)
    for __ in range(250):
        expected = reference_update(expected, rule)
        tracker.step()
        assert np.array_equal(tracker.cells, expected)
        assert tracker.num_alive == expected.sum()
        rows, cols = np.nonzero(expected)
        assert tracker.bbox == (rows.min(), rows.max(), cols.min(), cols.max())


def find_parent_candidates(
    prev_cells: np.ndarray, row: int, col: int
) -> list[tuple[int, int]]:
    return [
        (row + dy, col + dx)
        for dy, dx in it.product((-1, 0, 1), repeat=2)
        if prev_cells[row + dy, col + dx]
    ]


def find_deposit_site(t: int, time_dilation: int = 1):
    if t % time_dilation:
        return None
    return dstream.steady_algo.assign_storage_site(64, t // time_dilation)


def make_inherited_mask(t: int, time_dilation: int = 1) -> np.uint64:
    """Mask for bits not deposited at step t."""
    site = find_deposit_site(t, time_dilation)
    if site is None:
        return np.uint64((1 << 64) - 1)
    return np.uint64(((1 << 64) - 1) ^ (1 << (63 - site)))


@pytest.mark.parametrize("name", RULES)
@pytest.mark.parametrize("time_dilation", [1, 3])
def test_marker_inheritance_and_deposit(name: str, time_dilation: int):
    rule, seed_pattern = RULES[name]
    tracker = FastCATracker(
        rule,
        make_grid(seed_pattern),
        dstream.steady_algo,
        time_dilation=time_dilation,
    )
    for t in range(1, 201):
        prev_cells, prev_markers = tracker.cells.copy(), tracker.markers.copy()
        tracker.step()
        assert tracker.t == t
        site = find_deposit_site(t, time_dilation)
        keep = make_inherited_mask(t, time_dilation)
        deposited = []
        for row, col in np.argwhere(tracker.cells):
            marker = tracker.markers[row, col]
            candidates = find_parent_candidates(prev_cells, row, col)
            assert candidates
            assert any(
                prev_markers[r, c] & keep == marker & keep
                for r, c in candidates
            )
            if site is not None:
                deposited.append(int(marker >> np.uint64(63 - site)) & 1)
        if len(deposited) > 100:  # fresh differentia are fair coin flips
            assert 0.3 < np.mean(deposited) < 0.7


def test_parent_choice_uniform():
    rule, seed_pattern = RULES["34life"]
    tracker = FastCATracker(rule, make_grid(seed_pattern), dstream.steady_algo)
    tracker.step(120)
    counts = {k: np.zeros(k, dtype=int) for k in range(1, 10)}
    for t in range(121, 161):
        prev_cells, prev_markers = tracker.cells.copy(), tracker.markers.copy()
        tracker.step()
        keep = make_inherited_mask(t)
        for row, col in np.argwhere(tracker.cells):
            candidates = find_parent_candidates(prev_cells, row, col)
            markers = [prev_markers[r, c] & keep for r, c in candidates]
            if len(set(markers)) < len(markers):
                continue  # parent not identifiable
            (which,) = np.flatnonzero(
                np.array(markers) == tracker.markers[row, col] & keep
            )
            counts[len(candidates)][which] += 1

    num_tested = 0
    for k, observed in counts.items():
        if k > 1 and observed.sum() >= 50 * k:
            assert scipy_stats.chisquare(observed).pvalue > 1e-4
            num_tested += 1
    assert num_tested >= 3


def test_seed_decorrelates_markers():
    rule, seed_pattern = RULES["outlier"]
    first, second = (
        FastCATracker(
            rule, make_grid(seed_pattern), dstream.steady_algo, seed=seed
        ).step(300)
        for seed in (1, 2)
    )
    assert np.array_equal(first.cells, second.cells)
    alive = first.cells.astype(bool)
    assert (first.markers[alive] != second.markers[alive]).mean() > 0.9


@pytest.mark.parametrize("time_dilation", [1, 4])
def test_make_population_df(time_dilation: int):
    rule, seed_pattern = RULES["outlier"]
    tracker = FastCATracker(
        rule,
        make_grid(seed_pattern),
        dstream.steady_algo,
        time_dilation=time_dilation,
    )
    tracker.step(300)
    assert tracker.num_deposits == 300 // time_dilation + 1
    df = tracker.make_population_df()
    positions = np.argwhere(tracker.cells)
    assert len(df) == tracker.num_alive == len(positions)
    assert np.array_equal(df[["row", "col"]].to_numpy(), positions)
    assert df["extant"].all()
    assert (df["dstream_S"] == 64).all()
    assert (df["dstream_storage_bitwidth"] == 64).all()
    assert (df["dstream_algo"] == "dstream.steady_algo").all()

    # as packed by pylib.track_ca.MultiTracker with dstream_bitwidth=1
    for (row, col), data_hex in zip(positions, df["data_hex"]):
        marker = int(tracker.markers[row, col])
        site_bits = [(marker >> (63 - site)) & 1 for site in range(64)]
        expected = (
            np.asarray(tracker.num_deposits, dtype=">u4").tobytes().hex()
            + np.packbits(site_bits, bitorder="big").tobytes().hex()
        )
        assert data_hex == expected


def test_reaching_grid_edge_raises():
    rule, seed_pattern = RULES["34life"]
    tracker = FastCATracker(
        rule, make_grid(seed_pattern, 64), dstream.steady_algo
    )
    with pytest.raises(AssertionError, match="edge of the grid"):
        tracker.step(500)


@pytest.mark.parametrize("algo_name", ["steady_algo", "tilted_algo"])
@pytest.mark.parametrize("time_dilation", [1, 5])
def test_reported_deposits_match_surface(algo_name: str, time_dilation: int):
    # the differentia held at each site must be the one that downstream
    # says is there, given the number of deposits reported
    algo = getattr(dstream, algo_name)
    rule, seed_pattern = RULES["outlier"]
    tracker = FastCATracker(
        rule, make_grid(seed_pattern), algo, time_dilation=time_dilation
    )
    last_deposit = {0: 0}  # site -> index of deposit, as tracker makes them
    for t in range(1, 701):
        tracker.step()
        if t % time_dilation == 0:
            site = algo.assign_storage_site(64, t // time_dilation)
            if site is not None:
                last_deposit[site] = t // time_dilation
    expected = list(algo.lookup_ingest_times_eager(64, tracker.num_deposits))
    assert [last_deposit[site] for site in range(64)] == expected


def test_snapshot():
    rule, seed_pattern = RULES["outlier"]
    tracker = FastCATracker(rule, make_grid(seed_pattern), dstream.steady_algo)
    num_early = tracker.step(200).snapshot().num_alive
    df = tracker.step(100).reconstruct_phylogeny()
    assert (df["snapshot_step"] == 200).sum() == num_early
    assert (df["snapshot_step"] == 300).sum() == tracker.num_alive
    assert df["extant"].sum() == tracker.num_alive
    assert df.loc[df["extant"], "snapshot_step"].eq(300).all()
