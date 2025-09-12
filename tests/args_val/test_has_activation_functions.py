import re
from typing import Iterable
import pytest
import torch.nn as nn
from torch.nn import Module
from template_nn.args_val import has_activation_functions


@pytest.mark.parametrize("activation_functions", [nn.ReLU()])
def test_has_activation_functions_ok(activation_functions: Iterable[Module]):
    has_activation_functions(activation_functions)


@pytest.mark.parametrize("activation_functions", [[]])
def test_has_activation_functions_error(activation_functions: Iterable[Module]):
    with pytest.raises(
        ValueError, match=re.escape("No activation functions were provided.")
    ):
        has_activation_functions(activation_functions)
