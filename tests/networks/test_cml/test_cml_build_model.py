import torch.nn as nn
from template_nn.networks.cml import CML


def test_cml_build_model_structure():
    cml = CML(
        {"conv_channels": [3, 6, 16], "conv_kernel_size": 3, "pool_kernel_size": 2},
        visualise=False,
    )

    model = cml.model
    assert isinstance(model, nn.Sequential)
    assert len(model) == 4  # Two conv-pool blocks

    assert isinstance(model[0], nn.Conv2d)
    assert model[0].in_channels == 3
    assert model[0].out_channels == 6
    assert model[0].kernel_size == (3, 3)

    assert isinstance(model[1], nn.MaxPool2d)
    assert model[1].kernel_size == 2
    assert model[1].stride == 2

    assert isinstance(model[2], nn.Conv2d)
    assert model[2].in_channels == 6
    assert model[2].out_channels == 16
    assert model[2].kernel_size == (3, 3)

    assert isinstance(model[3], nn.MaxPool2d)
    assert model[3].kernel_size == 2
    assert model[3].stride == 2


def test_cml_build_model_single_block():
    cml = CML(
        {"conv_channels": [1, 10], "conv_kernel_size": 5, "pool_kernel_size": 3},
        visualise=False,
    )

    model = cml.model
    assert isinstance(model, nn.Sequential)
    assert len(model) == 2  # One conv-pool block

    assert isinstance(model[0], nn.Conv2d)
    assert model[0].in_channels == 1
    assert model[0].out_channels == 10
    assert model[0].kernel_size == (5, 5)

    assert isinstance(model[1], nn.MaxPool2d)
    assert model[1].kernel_size == 3
    assert model[1].stride == 2
