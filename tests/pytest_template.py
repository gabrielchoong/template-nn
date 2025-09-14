import pytest


def is_odd(n: int) -> bool:
    return n % 2 == 1


@pytest.mark.parametrize(
    "input_val, expected",
    [
        (1, True),
        (2, False),
        (3, True),
        (0, False),
        (-1, True),
        (-2, False),
    ],
)
def test_is_odd(input_val, expected):
    assert is_odd(input_val) == expected, (
        f"Expected is_odd({input_val}) to be {expected}"
    )
