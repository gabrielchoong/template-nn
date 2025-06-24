import pytest
from torch import nn
from template_nn._utils.layer_gen import create_layers


def test_create_layers_structure():
    layers = create_layers(input_size=10,
                           output_size=2,
                           hidden_sizes=[16, 8],
                           activation_functions=[nn.ReLU(),
                                                 nn.Tanh()])

    # Expect: Linear -> ReLU -> Linear -> Tanh -> Linear
    assert len(layers) == 5

    assert isinstance(layers[0], nn.Linear)
    assert layers[0].in_features == 10
    assert layers[0].out_features == 16

    assert isinstance(layers[1], nn.ReLU)

    assert isinstance(layers[2], nn.Linear)
    assert layers[2].in_features == 16
    assert layers[2].out_features == 8

    assert isinstance(layers[3], nn.Tanh)

    assert isinstance(layers[4], nn.Linear)
    assert layers[4].in_features == 8
    assert layers[4].out_features == 2


@pytest.mark.parametrize(
    "input_size, output_size, hidden_sizes, activations, expected_types", [
        (5, 1, [4], [nn.ReLU()], [nn.Linear, nn.ReLU, nn.Linear]),
        (3, 1, [3, 3], [nn.Tanh(), nn.Sigmoid()],
         [nn.Linear, nn.Tanh, nn.Linear, nn.Sigmoid, nn.Linear]),
    ])
def test_create_layers_parametrized(input_size, output_size, hidden_sizes,
                                    activations, expected_types):
    layers = create_layers(input_size=input_size,
                           output_size=output_size,
                           hidden_sizes=hidden_sizes,
                           activation_functions=activations)

    assert len(layers) == len(expected_types)
    for layer, expected_type in zip(layers, expected_types):
        assert isinstance(layer, expected_type)
