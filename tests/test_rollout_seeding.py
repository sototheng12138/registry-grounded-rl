import pytest

from registry_grounded_rl.rollout_seeding import independent_rollout_seed


def _seed(**overrides: int) -> int:
    values = {
        "base_seed": 2801,
        "global_step": 1,
        "round_index": 0,
        "rank": 0,
        "rollout_index": 0,
    }
    values.update(overrides)
    return independent_rollout_seed(**values)


def test_independent_rollout_seed_is_reproducible_and_bounded() -> None:
    assert _seed() == _seed()
    assert 0 <= _seed() < 2**31 - 1


def test_independent_rollout_seed_separates_rollout_coordinates() -> None:
    seeds = {
        independent_rollout_seed(
            base_seed=2801,
            global_step=step,
            round_index=round_index,
            rank=rank,
            rollout_index=rollout_index,
        )
        for step in range(3)
        for round_index in range(10)
        for rank in range(2)
        for rollout_index in range(96)
    }
    assert len(seeds) == 3 * 10 * 2 * 96


@pytest.mark.parametrize(
    "overrides,exception",
    [
        ({"base_seed": -1}, ValueError),
        ({"global_step": True}, TypeError),
    ],
)
def test_independent_rollout_seed_rejects_invalid_coordinates(
    overrides: dict[str, int], exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        _seed(**overrides)
