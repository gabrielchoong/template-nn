import re
import pytest
import torch.nn as nn
from template_nn._utils.args_val import activation_functions_check


@pytest.mark.parametrize("activation_functions, hidden_sizes",
                         [([nn.ReLU], [5]), ([nn.ReLU, nn.Tanh], [10, 4])])
def test_activation_functions_check_ok(activation_functions, hidden_sizes):
    activation_functions_check(activation_functions, hidden_sizes)


@pytest.mark.parametrize(
    "activation_functions, hidden_sizes",
    [([nn.ReLU()], [5, 5]),
     ([nn.ReLU(), nn.Tanh(), nn.Sigmoid()], [4, 4])])
def test_activation_functions_check_error(activation_functions, hidden_sizes):
    with pytest.raises(
            ValueError,
            match=re.escape(
                "Number of activation functions does not match number of hidden node counts."
                +
                f"Expected {activation_functions} of hidden nodes, but got {hidden_sizes} instead."
            )):
        activation_functions_check(activation_functions, hidden_sizes)
