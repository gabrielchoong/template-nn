import pytest
import pandas as pd
import torch.nn as nn
from template_nn.networks.base_nn import BaseNetwork


# A dummy subclass for testing BaseNetwork's __init__
class DummyNetwork(BaseNetwork):

    def __init__(self, tabular, model_keys, visualise=False):
        super().__init__(tabular, model_keys, visualise)

    def _create_layers(self, *args, **kwargs):
        # This will be called by BaseNetwork's __init__
        # We don't need to test its functionality here, just that it's called.
        return []

    def _build_model(self, *args, **kwargs):
        # This will be called by BaseNetwork's __init__
        # We don't need to test its functionality here, just that it's called.
        return nn.Sequential()


def test_base_network_init_dict():
    tabular_data = {"key1": 1, "key2": "value"}
    model_keys = ["key1", "key2"]
    net = DummyNetwork(tabular_data, model_keys)
    assert net.tabular == tabular_data
    assert net.model_keys == model_keys
    assert not net.visualise


def test_base_network_init_dataframe():
    tabular_data = pd.DataFrame({"key1": [1], "key2": ["value"]})
    model_keys = ["key1", "key2"]
    net = DummyNetwork(tabular_data, model_keys, visualise=True)
    assert net.tabular.equals(tabular_data)
    assert net.model_keys == model_keys
    assert net.visualise


def test_base_network_abstract_methods_raise_not_implemented_error_on_init():
    # A subclass that does NOT implement the abstract methods
    class IncompleteNetwork(BaseNetwork):

        def __init__(self, tabular, model_keys, visualise=False):
            super().__init__(tabular, model_keys, visualise)

    # Provide minimal valid data for __init__
    tabular_data = {"key1": 1, "key2": "value"}
    model_keys = ["key1", "key2"]

    with pytest.raises(NotImplementedError,
                       match="Define how model is built here"):
        IncompleteNetwork(tabular_data, model_keys)
