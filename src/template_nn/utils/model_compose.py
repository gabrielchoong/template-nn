from typing import Iterable

import pandas as pd
import torch.nn as nn

from template_nn.utils.args_val import validate_args
from template_nn.utils.layer_gen import create_layers
from template_nn.utils.model_compose_utils import get_params, warn_hidden_layer, sized_to_list


def build_model(input_size: int,
                output_size: int,
                hidden_sizes: Iterable[int],
                activation_functions: Iterable[nn.Module]) -> nn.Sequential:
    """
    A procedural function declaring the steps of building a model.
    :param input_size: The number of input features.
    :param output_size: The number of output features.
    :param hidden_sizes: The number of nodes in each hidden layer.
    :param activation_functions: The activation functions to use.
    :return: A torch.nn.Sequential object representing the layers.
    """

    # missing arguments will result in errors that are hard to debug
    input_size, output_size, hidden_sizes, activation_functions \
        = validate_args(input_size, output_size, hidden_sizes, activation_functions)

    warn_hidden_layer(len(hidden_sizes))

    hidden_sizes = sized_to_list(hidden_sizes)

    layers = create_layers(input_size, output_size, hidden_sizes, activation_functions)

    model = nn.Sequential(*layers)

    return model


def build_tabular_model(tabular: dict | pd.DataFrame) -> nn.Sequential:
    """
    Must contain keys of: ("input_size", "output_size", "hidden_sizes", "activation_functions")
    :param tabular: A dict or pandas DataFrame representing the tabular data.
    :return: A torch.nn.Sequential object representing the layers.
    """

    keys = ("input_size", "output_size", "hidden_sizes", "activation_functions")

    input_size, output_size, hidden_sizes, activation_functions = get_params(tabular, keys)

    return build_model(input_size, output_size, hidden_sizes, activation_functions)


def build_norm_model(input_size: int,
                     output_size: int,
                     hidden_sizes: Iterable[int],
                     activation_functions: Iterable[nn.Module]) -> nn.Sequential:
    """
    :param input_size: The number of input features.
    :param output_size: The number of output features.
    :param hidden_sizes: The number of nodes in each hidden layer.
    :param activation_functions: The activation functions to use.
    :return: A torch.nn.Sequential object representing the layers.
    """
    return build_model(input_size, output_size, hidden_sizes, activation_functions)
