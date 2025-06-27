import torch.nn as nn
from template_nn.networks.cnn import CNN


def test_compute_output_dim():
    config = {
        "image_size": (64, 64),
        "conv_channels": [3, 6, 16],
        "conv_kernel_size": 5,
        "pool_kernel_size": 2,
        "fcn_hidden_sizes": [120, 84],
        "activation_functions": [nn.ReLU(), nn.ReLU()],
        "output_channel": 10
    }

    cnn = CNN(tabular=config)

    # Expected manually:
    # After conv: (64 - 5) / 1 + 1 = 60
    # After pool: 60 / 2 = 30
    h, w = cnn._compute_output_dim(64, 64, 5, 2)
    assert (h, w) == (30, 30)
