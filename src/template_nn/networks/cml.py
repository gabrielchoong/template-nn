from typing import List
import torch.nn as nn

from ..args_val import (is_iterable, is_positive_int)

from ..retrieve_keys import get_model_keys
from .base_nn import BaseNetwork


class CML(BaseNetwork):

    def __init__(self, tabular, visualise):
        super().__init__(tabular, get_model_keys("CML"), visualise)

    def _build_model(self,
                     conv_channels: List[int],
                     conv_kernel_size: int = 3,
                     pool_kernel_size: int = 3) -> nn.Sequential:

        is_positive_int(conv_kernel_size)
        is_positive_int(pool_kernel_size)

        is_iterable(conv_channels)
        for channel_sizes in conv_channels:
            is_positive_int(channel_sizes)

        return nn.Sequential(*self._create_layers(
            conv_channels, conv_kernel_size, pool_kernel_size))

    def _create_layers(self,
                       conv_channels: List[int],
                       conv_kernel_size: int = 3,
                       pool_kernel_size: int = 3) -> List[nn.Module]:

        layers = []

        in_size = conv_channels[0]

        for out_size in conv_channels[1:]:
            layers += [
                nn.Conv2d(
                    in_channels=in_size,
                    out_channels=out_size,
                    kernel_size=conv_kernel_size,
                ),
                nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
            ]

            in_size = out_size

        return layers
