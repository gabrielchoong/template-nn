import pytest
from torch import nn
from template_nn import FNN


def test_create_layers_structure():
    model = FNN({
        "input_size": 10,
        "output_size": 2,
        "hidden_sizes": [16, 8],
        "activation_functions": [nn.ReLU(), nn.Tanh()]
    })
    net = model.model
    # Expect: Linear -> ReLU -> Linear -> Tanh -> Linear
    assert len(net) == 5

    assert isinstance(net[0], nn.Linear)
    assert net[0].in_features == 10
    assert net[0].out_features == 16

    assert isinstance(net[1], nn.ReLU)

    assert isinstance(net[2], nn.Linear)
    assert net[2].in_features == 16
    assert net[2].out_features == 8

    assert isinstance(net[3], nn.Tanh)

    assert isinstance(net[4], nn.Linear)
    assert net[4].in_features == 8
    assert net[4].out_features == 2


@pytest.mark.parametrize(
    "input_size, output_size, hidden_sizes, activations, expected_types", [
        (5, 1, [4], [nn.ReLU()], [nn.Linear, nn.ReLU, nn.Linear]),
        (3, 1, [3, 3], [nn.Tanh(), nn.Sigmoid()],
         [nn.Linear, nn.Tanh, nn.Linear, nn.Sigmoid, nn.Linear]),
    ])
def test_create_layers_parametrized(input_size, output_size, hidden_sizes,
                                    activations, expected_types):
    model = FNN({
        "input_size": input_size,
        "output_size": output_size,
        "hidden_sizes": hidden_sizes,
        "activation_functions": activations
    })
    net = model.model

    assert len(net) == len(expected_types)
    for layer, expected_type in zip(net, expected_types):
        assert isinstance(layer, expected_type)
