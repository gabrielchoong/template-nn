import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from ..args_val import is_dict, is_valid_keys


class BaseNetwork(nn.Module, ABC):
    """
    All network classes should inherit from this class. This class is not supposed to be constructed directly.
    """

    # cleanup `visualise` in a future release
    def __init__(self, visualise: bool) -> None:
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
        model_keys: tuple[str],
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
