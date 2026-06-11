from abc import ABC, abstractmethod
from typing import Iterable

import torch
from torch import nn


class BaseNetwork(nn.Module, ABC):
    """
    All network classes should inherit from this class. This class is not supposed to be constructed directly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential()

    # only override this function on rare occasions where `_build_model`
    # and `_create_layers` is insufficient for the model complexity
    # this is generally unnecessary and not recommended to be overriden
    def forward(self, x: torch.Tensor) -> nn.Module:
        return self.model(x)

    # stick to implementing `_build_model` and `_create_layers`
    # you do not need to overwrite this 99% of the time
    def _get_params(
        self,
        model_config: dict[str, int | list[int] | list[str]],
        model_keys: list[str],
    ) -> list:
        """
        Dynamically retrieve model specific parameters.
        """
        is_valid_keys(model_config, model_keys)

        return is_dict(model_config, model_keys)

    @abstractmethod
    def _build_model(self, *args, **kwargs) -> nn.Sequential:
        """
        Return a `nn.Sequential` object as the return value.

        Check all of the variables with functions defined in `args_val.py`.

        Recommended syntax:
        ```
            # verify arguments
            is_positive_int(arg)
            ...

            # build the model
            return nn.Sequential(*self._create_layers(*args, **kwargs))
        ```
        """
        raise NotImplementedError("Define how model is built here")

    @abstractmethod
    def _create_layers(self, *args, **kwargs) -> list[nn.Module]:
        """
        The logic for creating neural networks dynamically.

        Recommended syntax:
        ```
            layers: list[nn.Module] = []
            for foo in bar:
                layers.append(foo)
                ...
            return layers
        ```
        """
        raise NotImplementedError("Define layer structure here")


FNN_KEYS = [
    "input_size",
    "output_size",
    "hidden_sizes",
    "activation_functions",
]


class FNN(BaseNetwork):
    """
    A Feedforward Neural Network (FNN) model for supervised learning.
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
        super().__init__()
        self.model_keys = FNN_KEYS
        self.params = self._get_params(model_config, self.model_keys)
        self.model = self._build_model(*self.params)

        print(self) if visualise else None

    def _build_model(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int],
        activation_functions: list[str],
    ) -> nn.Sequential:
        is_positive_int(input_size)
        is_positive_int(output_size)
        is_iterable(hidden_sizes)

        for sizes in hidden_sizes:
            is_positive_int(sizes)

        has_activation_functions(activation_functions)
        activation_functions_check(activation_functions, hidden_sizes)

        return nn.Sequential(
            *self._create_layers(
                input_size, output_size, hidden_sizes, activation_functions
            )
        )

    def _create_layers(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int],
        activation_functions: list[str],
    ) -> list[nn.Module]:
        layers: list[nn.Module] = []
        in_size = input_size

        for hidden_size, activation_function in zip(hidden_sizes, activation_functions):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(getattr(nn, activation_function)())
            in_size = hidden_size

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        return layers


CML_KEYS = [
    "conv_channels",
    "conv_kernel_size",
    "pool_kernel_size",
]


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
        super().__init__()
        self.model_keys = CML_KEYS
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


CNN_KEYS = [
    "image_size",
    "conv_channels",
    "conv_kernel_size",
    "pool_kernel_size",
    "fcn_hidden_sizes",
    "activation_functions",
    "output_channel",
]


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
        super().__init__()
        self.model_keys = CNN_KEYS
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


__all__ = ["FNN", "CNN"]
