import torch.nn as nn

from ..args_val import is_iterable, is_positive_int

from ..retrieve_keys import get_model_keys
from .base_nn import BaseNetwork


class CML(BaseNetwork):
    """
    A Convolution - MaxPooling Layer (CML) component for CNNs.
    """

    def __init__(
        self,
        model_config: dict[str, int | list[int] | list[str]],
        visualise: bool = False,
    ) -> None:
        """
        :params model_config: A dictionary / json-like structure for model configuration
        :params visualise: A boolean type for visualising structure. Default (False).
        """
        super().__init__(visualise)
        self.model_keys = get_model_keys("CML")
        self.params = self._get_params(model_config, self.model_keys)
        self.model = self._build_model(*self.params)

        print(self) if visualise else None

    def _build_model(
        self,
        conv_channels: list[int],
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 3,
    ) -> nn.Sequential:
        is_positive_int(conv_kernel_size)
        is_positive_int(pool_kernel_size)
        is_iterable(conv_channels)

        for channel_sizes in conv_channels:
            is_positive_int(channel_sizes)

        return nn.Sequential(
            *self._create_layers(conv_channels, conv_kernel_size, pool_kernel_size)
        )

    def _create_layers(
        self,
        conv_channels: list[int],
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 3,
    ) -> list[nn.Module]:
        layers: list[nn.Module] = []
        stride = 2
        in_size = conv_channels[0]

        for out_size in conv_channels[1:]:
            layers.append(nn.Conv2d(in_size, out_size, conv_kernel_size))
            layers.append(nn.MaxPool2d(pool_kernel_size, stride))
            in_size = out_size

        return layers
