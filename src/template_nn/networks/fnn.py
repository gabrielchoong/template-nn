import torch.nn as nn

from .base_nn import BaseNetwork
from ..retrieve_keys import get_model_keys
from ..args_val import (
    is_positive_int,
    is_iterable,
    has_activation_functions,
    activation_functions_check,
)


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
        super().__init__(visualise)
        self.model_keys = get_model_keys("FNN")
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
