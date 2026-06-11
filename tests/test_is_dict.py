from template_nn import is_dict


def test_is_dict_returns_expected_values():
    tabular = {"a": 1, "b": 2}
    assert is_dict(tabular, ("a", "b")) == [1, 2]
