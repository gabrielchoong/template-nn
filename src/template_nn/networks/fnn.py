from typing import List

import pandas as pd
import torch.nn as nn

from .base_nn import BaseNetwork
from ..retrieve_keys import get_model_keys
from ..args_val import (is_positive_int, is_iterable, has_activation_functions,
                        activation_functions_check)


class FNN(BaseNetwork):
    """
    A Feedforward Neural Network (FNN) model for supervised learning.

    References:
        - Suganthan, P. N., & Katuwal, R. (2021). On the origins of randomization-based feedforward neural networks.
          *Applied Soft Computing*, 105, 107239. [DOI: 10.1016/j.asoc.2021.107239](https://doi.org/10.1016/j.asoc.2021.107239)

    """

    def __init__(self,
                 tabular: dict | pd.DataFrame,
                 visualise: bool = False) -> None:
        """
        Class description goes here
        :param tabular: An input accepting both a dictionary or a pandas.DataFrame object.
        :param visualise: A toggle switch to visualize the model. OFF(False) by default.
        """

        super().__init__(tabular, get_model_keys("FNN"), visualise)

    def _build_model(self, input_size: int, output_size: int,
                     hidden_sizes: List[int],
                     activation_functions: List[nn.Module]) -> nn.Sequential:
        """
        A procedural function declaring the steps of building a model.
        :param input_size: The number of input features.
        :param output_size: The number of output features.
        :param hidden_sizes: The number of nodes in each hidden layer.
        :param activation_functions: The activation functions to use.
        :return: A torch.nn.Sequential object representing the layers.
        """

        is_positive_int(input_size)
        is_positive_int(output_size)
        is_iterable(hidden_sizes)

        for sizes in hidden_sizes:
            is_positive_int(sizes)

        has_activation_functions(activation_functions)
        activation_functions_check(activation_functions, hidden_sizes)

        return nn.Sequential(*self._create_layers(
            input_size, output_size, hidden_sizes, activation_functions))

    def _create_layers(
            self, input_size: int, output_size: int, hidden_sizes: List[int],
            activation_functions: List[nn.Module]) -> list[nn.Module]:
        """
        A function to generate layers dynamically.
        :param input_size: The number of input features.
        :param hidden_sizes: The number of nodes in each hidden layer.
        :param output_size: The number of output features.
        :param activation_functions: The activation functions to use.
        :return: A list of torch.nn activation functions. Refer to the official documentation for possible inputs:
        https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity and
        https://pytorch.org/docs/stable/nn.html#non-linear-activations-other
        """

        layers = []

        in_size = input_size

        for hidden_size, activation_function in zip(hidden_sizes,
                                                    activation_functions):
            layers += [nn.Linear(in_size, hidden_size), activation_function]

            # go to next layer
            in_size = hidden_size

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        return layers
