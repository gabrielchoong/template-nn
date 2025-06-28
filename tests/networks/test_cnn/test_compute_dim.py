import pytest
import torch.nn as nn
from template_nn.networks.cnn import CNN


@pytest.mark.parametrize(
    "in_dim, kernel_size, stride, padding, dilation, expected",
    [
        (32, 5, 1, 0, 1, 28),  # classic conv
        (28, 2, 2, 0, 1, 14),  # pooling
        (28, 3, 2, 1, 1, 14),  # padding applied
        (64, 5, 1, 2, 1, 64),  # dilation=1, padding=2
    ])
def test_compute_dim(in_dim, kernel_size, stride, padding, dilation, expected):
    config = {
        "image_size": (64, 64),
        "conv_channels": [3, 6, 16],
        "conv_kernel_size": 3,
        "pool_kernel_size": 2,
        "fcn_hidden_sizes": [120, 80],
        "activation_functions": [nn.ReLU(), nn.ReLU()],
        "output_channel": 10
    }
    cnn = CNN(tabular=config)
    result = cnn._compute_dim(in_dim, kernel_size, stride, padding, dilation)
    assert result == expected
