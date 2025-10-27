import itertools as it
import logging
import os
import sys

import msprime
import numpy as np
import pandas as pd
from tqdm import tqdm


def build_demography(
    grid_dim: int,
    deme_size: int,
    migration_rate: float,
) -> msprime.Demography:
    def get_deme_id(i: int, j: int) -> int:
        """Converts (row, col) to a 0-indexed deme ID."""
        return i * grid_dim + j

    demography = msprime.Demography()
    for i in range(grid_dim):
        for j in range(grid_dim):
            deme_name = f"deme_{i}_{j}"
            demography.add_population(name=deme_name, initial_size=deme_size)

    migration_matrix = [[0.0] * num_demes for _ in range(num_demes)]

    for i1, j1 in it.product(
        range(grid_dim),
        repeat=2,
    ):
        deme_id_1 = get_deme_id(i1, j1)
        neighbors = []
        if i1 + 1 < grid_dim:  # neighbor below
            neighbors.append((i1 + 1, j1))
        if j1 + 1 < grid_dim:  # neighbor to the right
            neighbors.append((i1, j1 + 1))

        for i2, j2 in neighbors:
            deme_id_2 = get_deme_id(i2, j2)
            migration_matrix[deme_id_1][deme_id_2] = migration_rate
            migration_matrix[deme_id_2][deme_id_1] = migration_rate

    demography.migration_matrix = np.array(migration_matrix)

    return demography


if __name__ == "__main__":
    logging.basicConfig(
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )
    if not sys.__stdout__.isatty():
        os.environ["TQDM_MININTERVAL"] = "5"

    logging.info(f"{os.getcwd()=}")
    logging.info(f"{sys.argv=}")

    slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", -1))
    logging.info(f"{slurm_array_task_id=}")

    grid_dim = int(sys.argv[1])
    num_alleles = 512
    ploidy = 1
    num_demes = grid_dim * grid_dim
    deme_size = num_alleles // ploidy
    migration_rate = 1 / 32  # 5%
    random_seed = int(os.getenv("SLURM_ARRAY_TASK_ID", 0)) + 1

    config = {
        "grid_dim": grid_dim,
        "num_alleles": num_alleles,
        "num_demes": num_demes,
        "deme_size": deme_size,
        "migration_rate": migration_rate,
        "ploidy": ploidy,
        "random_seed": random_seed,
        "slurm_array_task_id": slurm_array_task_id,
    }
    logging.info(f"{config=}")

    demography = build_demography(
        grid_dim=grid_dim,
        deme_size=deme_size,
        migration_rate=migration_rate,
    )
    sample_sets = [
        msprime.SampleSet(num_samples=deme_size, population=i)
        for i in range(num_demes)
    ]

    logging.info("running sim_ancestry...")
    ts = msprime.sim_ancestry(
        additional_nodes=msprime.NodeType.MIGRANT,
        coalescing_segments_only=False,
        demography=demography,
        ploidy=ploidy,
        random_seed=random_seed,
        samples=sample_sets,
    )
    logging.info("... done!")
    logging.info(f"{ts.num_trees} trees, {ts.num_nodes} total nodes")

    logging.info("collating edge data...")
    edge_df = pd.DataFrame.from_records(
        [
            {
                "id": edge.child,
                "population": ts.node(edge.child).population,
                "origin_time": ts.node(edge.child).time,
                "ancestor_id": edge.parent,
                "ancestor_population": ts.node(edge.parent).population,
                "ancestor_time": ts.node(edge.parent).time,
                **config,
            }
            for edge in tqdm(ts.edges())
        ],
    )
    logging.info("... done!")
    logging.info(f"edge_df shape: {edge_df.shape}")
    logging.info(f"{edge_df.describe()=}")
    logging.info(f"{edge_df.head()=}")

    filename = f"a=edges+{slurm_array_task_id=}+ext=.pqt"
    logging.info(f"saving edge_df to {filename}...")
    edge_df.to_parquet(filename)
    logging.info("... done!")

    logging.info("tabulating migration counts by population...")
    migrations_df = edge_df.loc[
        edge_df["population"] != edge_df["ancestor_population"],
        "population",
    ].value_counts()
    traffic_df = (
        migrations_df.rename_axis("population")
        .reset_index(name="migration_count")
        .assign(**config)
    )
    logging.info("... done!")
    logging.info(f"traffic_df shape: {traffic_df.shape}")
    logging.info(f"{traffic_df.describe()=}")
    logging.info(f"{traffic_df.head()=}")
    assert len(traffic_df) == num_demes

    filename = f"a=traffic+{slurm_array_task_id=}+ext=.pqt"
    logging.info(f"saving traffic_df to {filename}...")
    traffic_df.to_parquet(filename)
    logging.info("... done!")

    logging.info("tabulating duration by population...")
    duration_df = (
        edge_df.groupby("population")["origin_time"]
        .max(
            "origin_time",
        )
        .reset_index()
        .assign(**config)
    )
    logging.info("... done!")
    logging.info(f"duration_df shape: {duration_df.shape}")
    logging.info(f"{duration_df.describe()=}")
    logging.info(f"{duration_df.head()=}")
    assert len(duration_df) == num_demes

    filename = f"a=duration+{slurm_array_task_id=}+ext=.pqt"
    logging.info(f"saving duration_df to {filename}...")
    traffic_df.to_parquet(filename)
    logging.info("... done!")
