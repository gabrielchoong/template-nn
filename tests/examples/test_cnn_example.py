import torch.nn as nn
from template_nn.examples.cnn_example import create_cnn_model


def test_cnn_example_creates_model():
    model = create_cnn_model()
    assert isinstance(model, nn.Module)  # CNN returns a nn.Module
    assert hasattr(model,
                   'model')  # Check if it has the internal sequential model
    assert isinstance(model.model, nn.Sequential)
