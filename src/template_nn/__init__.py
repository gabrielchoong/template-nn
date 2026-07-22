from abc import ABC, abstractmethod
from typing import TypedDict

import torch
from torch import nn


KEYS = {  # TODO: Is it possible to refactor such that this dict is no longer needed?
    "FNN": [
        "input_size",
        "output_size",
        "hidden_sizes",
        "activation_functions",
    ],
    "CNN": [
        "image_size",
        "conv_channels",
        "conv_kernel_size",
        "pool_kernel_size",
        "fcn_hidden_sizes",
        "activation_functions",
        "output_channel",
    ],
    "CML": [
        "conv_channels",
        "conv_kernel_size",
        "pool_kernel_size",
    ],
}


class Config(TypedDict):
    pass


class FNNConfig(Config):
    input_size: int
    output_size: int
    hidden_sizes: list[int]
    activation_functions: list[str]


class CNNConfig(Config):
    image_size: tuple[int, int]
    conv_channels: list[int]
    conv_kernel_size: int
    pool_kernel_size: int
    fcn_hidden_sizes: list[int]
    activation_functions: list[str]
    output_channel: int


class CMLConfig(Config):
    conv_channels: list[int]
    pool_method: str
    conv_kernel_size: int
    pool_kernel_size: int


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
        try:
            return self.model(x)
        except Exception as e:
            raise e

    # stick to implementing `_build_model` and `_create_layers`
    # you do not need to overwrite this 99% of the time
    def _get_params(
        self,
        model_config: Config,
        model_keys: list[str],
    ) -> list:
        """
        Dynamically retrieve model specific parameters.
        """
        try:
            # It's been too long since I worked on this codebase hence I barely
            # know what this one-liner was initially supposed to be.
            # It was refactored from the `is_dict` function which could be
            # rewritten as the one-liner below.
            # It is probably to load the actual configuration defined by the
            # user and turning the values into a list which is to be processed
            # further down the data pipeline.
            return [model_config[key] for key in model_keys]
        except Exception as e:
            raise e

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


class FNN(BaseNetwork):
    """
    A Feedforward Neural Network (FNN) model for supervised learning.
    """

    def __init__(
        self,
        model_config: FNNConfig,
        visualise: bool = False,
    ) -> None:
        """
        :params model_config: A dictionary / json-like structure for model configuration
        :params visualise: A boolean type for visualising structure. Default (False).
        """
        super().__init__()
        self.model_keys = KEYS["FNN"]
        self.params = self._get_params(model_config, self.model_keys)
        self.model = self._build_model(*self.params)

        print(self) if visualise else None

    def _build_model(self, *kwargs) -> nn.Sequential:
        try:
            return nn.Sequential(*self._create_layers(*kwargs))
        except Exception as e:
            raise e

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


class CML(BaseNetwork):
    """
    A Convolution - MaxPooling Layer (CML) component for CNNs.
    """

    def __init__(
        self,
        model_config: CMLConfig,
        visualise: bool = False,
    ) -> None:
        """
        :params model_config: A dictionary / json-like structure for model configuration
        :params visualise: A boolean type for visualising structure. Default (False).
        """
        super().__init__()
        self.model_keys = KEYS["CML"]
        self.params = self._get_params(model_config, self.model_keys)
        self.model = self._build_model(*self.params)

        print(self) if visualise else None

    def _build_model(self, *kwargs) -> nn.Sequential:
        try:
            return nn.Sequential(*self._create_layers(*kwargs))
        except Exception as e:
            raise e

    def _create_layers(
        self,
        conv_channels: list[int],
        pool_method: str = "MaxPool2d",
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 3,
    ) -> list[nn.Module]:
        layers: list[nn.Module] = []
        stride = 2
        in_size = conv_channels[0]

        for out_size in conv_channels[1:]:
            layers.append(nn.Conv2d(in_size, out_size, conv_kernel_size))
            layers.append(getattr(nn, pool_method)(pool_kernel_size, stride))
            in_size = out_size

        return layers


class CNN(BaseNetwork):
    """
    A Convolutional Neural Network (CNN) model for supervised learning.
    """

    def __init__(
        self,
        model_config: CNNConfig,
        visualise: bool = False,
    ) -> None:
        """
        :params model_config: A dictionary / json-like structure for model configuration
        :params visualise: A boolean type for visualising structure. Default (False).
        """
        super().__init__()
        self.model_keys = KEYS["CNN"]
        self.params = self._get_params(model_config, self.model_keys)
        self.model = self._build_model(*self.params)

        print(self) if visualise else None

    def _build_model(self, *kwargs) -> nn.Sequential:
        try:
            return nn.Sequential(*self._create_layers(*kwargs))
        except Exception as e:
            raise e

    def _create_layers(
        self,
        image_size: tuple[int, int],
        conv_channels: list[int],
        pool_method: str,
        conv_kernel_size: int,
        pool_kernel_size: int,
        fcn_hidden_sizes: list[int],
        activation_functions: list[str],
        output_channel: int,
    ) -> list[nn.Module]:
        height, width = image_size

        for _ in range(len(conv_channels) - 1):
            height, width = self._compute_output_dim(
                height, width, conv_kernel_size, pool_kernel_size
            )

        conv_layers = CML(
            {
                "conv_channels": conv_channels,
                "pool_method": pool_method,
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
