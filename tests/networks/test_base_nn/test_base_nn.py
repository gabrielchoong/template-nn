import pytest
import torch.nn as nn
from template_nn.networks.base_nn import BaseNetwork


# A dummy subclass for testing BaseNetwork's __init__
class DummyNetwork(BaseNetwork):
    def __init__(
        self,
        model_config: dict[str, int | list[int] | list[str]],
        visualise: bool = False,
    ) -> None:
        super().__init__(visualise)
        self.model_config = model_config

    def _create_layers(self, *args, **kwargs):
        # This will be called by BaseNetwork's __init__
        # We don't need to test its functionality here, just that it's called.
        return []

    def _build_model(self, *args, **kwargs):
        # This will be called by BaseNetwork's __init__
        # We don't need to test its functionality here, just that it's called.
        return nn.Sequential()


def test_base_network_init_dict():
    model_config_data = {"key1": 1, "key2": "value"}
    net = DummyNetwork(model_config_data, visualise=False)
    assert net.model_config == model_config_data
    assert not net.visualise


def test_base_network_abstract_methods_raise_not_implemented_error_on_init():
    # A subclass that does NOT implement the abstract methods
    class IncompleteNetwork(BaseNetwork):
        def __init__(
        self,
        visualise: bool = False,
    ) -> None:
            super().__init__(visualise)

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteNetwork(visualise=False) # type: ignore => this error is on purpose
