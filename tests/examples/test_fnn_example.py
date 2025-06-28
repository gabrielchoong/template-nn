import torch.nn as nn
from template_nn.examples.fnn_example import create_fnn_model


def test_fnn_example_creates_model():
    model = create_fnn_model()
    assert isinstance(model, nn.Module)  # FNN returns a nn.Module
    assert hasattr(model,
                   'model')  # Check if it has the internal sequential model
    assert isinstance(model.model, nn.Sequential)
