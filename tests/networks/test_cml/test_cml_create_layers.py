import torch.nn as nn
from template_nn.networks.cml import CML


def test_cml_create_layers_structure():
    cml_instance = CML(
        {
            "conv_channels": [3, 6, 16],
            "conv_kernel_size": 3,
            "pool_kernel_size": 2
        },
        visualise=False,
    )

    layers = cml_instance._create_layers(conv_channels=[3, 6, 16],
                                         conv_kernel_size=3,
                                         pool_kernel_size=2)
    assert isinstance(layers, list)
    assert len(layers) == 4  # Two conv-pool blocks

    assert isinstance(layers[0], nn.Conv2d)
    assert layers[0].in_channels == 3
    assert layers[0].out_channels == 6
    assert layers[0].kernel_size == (3, 3)

    assert isinstance(layers[1], nn.MaxPool2d)
    assert layers[1].kernel_size == 2
    assert layers[1].stride == 2

    assert isinstance(layers[2], nn.Conv2d)
    assert layers[2].in_channels == 6
    assert layers[2].out_channels == 16
    assert layers[2].kernel_size == (3, 3)

    assert isinstance(layers[3], nn.MaxPool2d)
    assert layers[3].kernel_size == 2
    assert layers[3].stride == 2


def test_cml_create_layers_single_block():
    cml_instance = CML(
        {
            "conv_channels": [1, 10],
            "conv_kernel_size": 5,
            "pool_kernel_size": 3
        },
        visualise=False,
    )

    layers = cml_instance._create_layers(conv_channels=[1, 10],
                                         conv_kernel_size=5,
                                         pool_kernel_size=3)
    assert isinstance(layers, list)
    assert len(layers) == 2  # One conv-pool block

    assert isinstance(layers[0], nn.Conv2d)
    assert layers[0].in_channels == 1
    assert layers[0].out_channels == 10
    assert layers[0].kernel_size == (5, 5)

    assert isinstance(layers[1], nn.MaxPool2d)
    assert layers[1].kernel_size == 3
    assert layers[1].stride == 2
