import pandas as pd

from pylib.persistence_filter import mark_persistent_tips


def make_df(ancestor_ids: list, early: set, late: set) -> pd.DataFrame:
    ids = range(len(ancestor_ids))
    return pd.DataFrame(
        {
            "id": ids,
            "ancestor_id": ancestor_ids,
            "label": ids,  # ids may be reassigned
            "is_early": [i in early for i in ids],
            "is_late": [i in late for i in ids],
        }
    )


def find_persistent(df: pd.DataFrame) -> list:
    res = mark_persistent_tips(df, early="is_early", late="is_late")
    return sorted(res.loc[res["is_persistent"], "label"])


def test_one_tip_per_persisting_lineage():
    #        0
    #      /   \
    #     1     2
    #    /|\   / \
    #   3 4 5 6   7
    #       |     |
    #      8,9    10
    # early tips: 3, 4 (siblings of the lineage leading to late 8, 9),
    # 6 (sibling of lineage leading to late 10)
    df = make_df(
        [0, 0, 0, 1, 1, 1, 2, 2, 5, 5, 7],
        early={3, 4, 6},
        late={8, 9, 10},
    )
    persistent = find_persistent(df)
    assert len(persistent) == 2
    assert persistent[0] in (3, 4) and persistent[1] == 6


def test_extinct_early_lineages_dropped():
    #      0
    #     / \
    #    1   2
    #   / \  |\
    #  3   4 5 6
    # early clade under 1 leaves no late descendants
    df = make_df([0, 0, 0, 1, 1, 2, 2], early={3, 4, 5}, late={6})
    assert find_persistent(df) == [5]


def test_early_tip_nested_under_early_only_node():
    #      0
    #     / \
    #    1   2
    #   / \   \
    #  3   4   5
    # early tips share an inner node that the late lineage lacks
    df = make_df([0, 0, 0, 1, 1, 2], early={3, 4}, late={5})
    persistent = find_persistent(df)
    assert len(persistent) == 1 and persistent[0] in (3, 4)


def test_distant_early_tip_not_marked():
    #        0
    #       / \
    #      1   2
    #     / \
    #    3   4
    #   / \
    #  5   6
    # early 2 is further from late 6 than early 5 is
    df = make_df([0, 0, 0, 1, 1, 3, 3], early={2, 4, 5}, late={6})
    assert find_persistent(df) == [5]


def test_no_late_tips():
    df = make_df([0, 0, 0], early={1, 2}, late=set())
    assert find_persistent(df) == []
