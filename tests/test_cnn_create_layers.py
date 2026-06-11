import torch.nn as nn
from template_nn import CNN


def test_create_layers_structure():
    model = CNN(
        {
            "image_size": (64, 64),
            "conv_channels": [3, 6, 16],
            "conv_kernel_size": 3,
            "pool_kernel_size": 2,
            "fcn_hidden_sizes": [120, 80],
            "activation_functions": ["ReLU", "ReLU"],
            "output_channel": 10,
        }
    )

    net = model.model
    assert isinstance(net, nn.Sequential)
    assert len(net) == 3  # CML, Flatten, FNN

    cml = net[0].model
    assert isinstance(cml, nn.Sequential)
    assert isinstance(cml[0], nn.Conv2d)
    assert cml[0].in_channels == 3
    assert cml[0].out_channels == 6

    assert isinstance(cml[1], nn.MaxPool2d)

    assert isinstance(cml[2], nn.Conv2d)
    assert cml[2].in_channels == 6
    assert cml[2].out_channels == 16

    assert isinstance(cml[3], nn.MaxPool2d)

    fnn = net[2].model
    assert isinstance(fnn, nn.Sequential)

    # implicit calculation for in_features
    assert isinstance(fnn[0], nn.Linear)
    assert fnn[0].out_features == 120

    assert isinstance(fnn[1], nn.ReLU)

    assert isinstance(fnn[2], nn.Linear)
    assert fnn[2].in_features == 120
    assert fnn[2].out_features == 80

    assert isinstance(fnn[3], nn.ReLU)

    assert isinstance(fnn[4], nn.Linear)
    assert fnn[4].in_features == 80
    assert fnn[4].out_features == 10
