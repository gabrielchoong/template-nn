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
