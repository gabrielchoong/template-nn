import re
import pytest
from template_nn.args_val import is_positive_int


@pytest.mark.parametrize("number", [1, 42, 1000, 99999, 100000])
def test_is_positive_int_ok(number):
    is_positive_int(number)


@pytest.mark.parametrize(
    "number", [-1, -100000, 0, 0.5, -0.0001, None, "string", {}, []]
)
def test_is_positive_int_error(number):
    with pytest.raises(
        ValueError, match=re.escape(f"Expected positive integer, got {number} instead.")
    ):
        is_positive_int(number)
