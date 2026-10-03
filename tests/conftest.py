import pandas as pd
import pytest

from recl2bench.rerankers.base import CandidateItem, ProfileItem, UserProfile

MS = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)  # noqa: E731


@pytest.fixture
def profile():
    return UserProfile("u1", "2022-07-16",
                       liked=(ProfileItem("i1", "Zelda", 5, ("Switch", "Adventure")),),
                       disliked=(), text="likes Zelda on Switch")


@pytest.fixture
def candidates():
    return [CandidateItem(f"c{i}", i, 1.0 - i / 10, f"item {i}") for i in range(5)]


@pytest.fixture
def ratings():
    """Tiny synthetic ratings table in Amazon'23 shape."""
    rows = [
        # u1: 3 prior, 2 positives in test window
        ("u1", "a", 5, "2020-01-01"), ("u1", "b", 4, "2021-01-01"), ("u1", "c", 2, "2022-01-01"),
        ("u1", "d", 5, "2022-08-01"), ("u1", "e", 4, "2023-01-01"),
        # u2: only 2 prior -> excluded
        ("u2", "a", 5, "2021-01-01"), ("u2", "b", 5, "2022-01-01"), ("u2", "d", 5, "2022-09-01"),
        # u3: 3 prior, test-window rating is 3 -> no positive -> excluded
        ("u3", "a", 4, "2020-01-01"), ("u3", "b", 4, "2020-02-01"), ("u3", "c", 4, "2020-03-01"),
        ("u3", "e", 3, "2022-10-01"),
        # items f/g appear only later (availability)
        ("u4", "f", 5, "2023-02-01"),
    ]
    return pd.DataFrame([(u, i, r, MS(t)) for u, i, r, t in rows],
                        columns=["user_id", "parent_asin", "rating", "timestamp"])
