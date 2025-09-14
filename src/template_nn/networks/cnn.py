import torch.nn as nn

from ..args_val import is_iterable, is_positive_int

from .base_nn import BaseNetwork
from .cml import CML
from .fnn import FNN
from ..retrieve_keys import get_model_keys


class CNN(BaseNetwork):
    """
    A Convolutional Neural Network (CNN) model for supervised learning.
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
        self.model_keys = get_model_keys("CNN")
        self.params = self._get_params(model_config, self.model_keys)
        self.model = self._build_model(*self.params)
        self.image_size = (0, 0)

        print(self) if visualise else None

    def _build_model(
        self,
        image_size: tuple[int, int],
        conv_channels: list[int],
        conv_kernel_size: int,
        pool_kernel_size: int,
        fcn_hidden_sizes: list[int],
        activation_functions: list[str],
        output_channel: int,
    ) -> nn.Sequential:
        self.image_size = image_size
        is_iterable(self.image_size)

        for sizes in self.image_size:
            is_positive_int(sizes)

        return nn.Sequential(
            *self._create_layers(
                conv_channels,
                conv_kernel_size,
                pool_kernel_size,
                fcn_hidden_sizes,
                activation_functions,
                output_channel,
            )
        )

    def _create_layers(
        self,
        conv_channels: list[int],
        conv_kernel_size: int,
        pool_kernel_size: int,
        fcn_hidden_sizes: list[int],
        activation_functions: list[str],
        output_channel: int,
    ) -> list[nn.Module]:
        height, width = self.image_size

        for _ in range(len(conv_channels) - 1):
            height, width = self._compute_output_dim(
                height, width, conv_kernel_size, pool_kernel_size
            )

        conv_layers = CML(
            {
                "conv_channels": conv_channels,
                "conv_kernel_size": conv_kernel_size,
                "pool_kernel_size": pool_kernel_size,
            },
            visualise=False,
        )

        fcn_layers = FNN(
            {
                "input_size": conv_channels[-1] * height * width,  # temp
                "hidden_sizes": fcn_hidden_sizes,
                "output_size": output_channel,
                "activation_functions": activation_functions,
            },
            visualise=False,
        )

        return [conv_layers, nn.Flatten(), fcn_layers]

    def _compute_dim(
        self, in_dim: int, kernel_size: int, stride=1, padding=0, dilation=1
    ) -> int:
        return (in_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def _compute_output_dim(
        self, height: int, width: int, conv_kernel_size: int, pool_kernel_size: int
    ) -> tuple[int, int]:
        height = self._compute_dim(height, conv_kernel_size)
        width = self._compute_dim(width, conv_kernel_size)

        height = self._compute_dim(height, pool_kernel_size, stride=pool_kernel_size)
        width = self._compute_dim(width, pool_kernel_size, stride=pool_kernel_size)

        return (height, width)
