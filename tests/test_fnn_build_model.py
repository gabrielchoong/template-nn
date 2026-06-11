import torch.nn as nn
from template_nn import FNN


def test_build_model_returns_sequential():
    model = FNN(
        {
            "input_size": 4,
            "output_size": 1,
            "hidden_sizes": [8, 4],
            "activation_functions": ["ReLU", "Tanh"],
        }
    )

    net = model.model
    assert isinstance(net, nn.Sequential)
    assert len(net) == 5
    assert isinstance(net[0], nn.Linear)
    assert isinstance(net[1], nn.ReLU)
    assert isinstance(net[2], nn.Linear)
    assert isinstance(net[3], nn.Tanh)
    assert isinstance(net[4], nn.Linear)
