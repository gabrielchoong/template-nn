import re
import pytest
from template_nn.args_val import is_iterable


@pytest.mark.parametrize(
    "number", [[1, 2, 3, 4, 5], [0, -1, -2], [100, 200], [5], [42, 43]]
)
def test_is_iterable_ok(number):
    is_iterable(number)


@pytest.mark.parametrize("number", [[], {}, set(), ""])
def test_is_iterable_empty_ok(number):
    is_iterable(number)


@pytest.mark.parametrize(
    "number",
    [42, None, 3.14],
)
def test_is_iterable_error(number):
    with pytest.raises(
        ValueError,
        match=re.escape(f"Expected iterable structure, got {number} instead."),
    ):
        is_iterable(number)
