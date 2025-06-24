import pytest

from template_nn._utils.model_compose import is_valid_keys


def test_is_valid_keys_raises():
    tabular = {"input_size": 10}
    with pytest.raises(ValueError, match="Tabular data must contain keys"):
        is_valid_keys(tabular, ("input_size", "output_size"))
