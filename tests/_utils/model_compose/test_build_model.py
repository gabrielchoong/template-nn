import torch.nn as nn
from template_nn._utils.model_compose import build_model


def test_build_model_returns_sequential():
    model = build_model(input_size=4,
                        output_size=1,
                        hidden_sizes=[8, 4],
                        activation_functions=[nn.ReLU(), nn.Tanh()])
    assert isinstance(model, nn.Sequential)
    assert len(model) == 5
    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[1], nn.ReLU)
    assert isinstance(model[2], nn.Linear)
    assert isinstance(model[3], nn.Tanh)
    assert isinstance(model[4], nn.Linear)
