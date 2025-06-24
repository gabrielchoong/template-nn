from typing import Iterable, Sized

import numpy as np
import pandas as pd
import torch.nn as nn

from .._utils.args_val import validate_args
from .._utils.layer_gen import create_layers


def build_model(input_size: int, output_size: int,
                hidden_sizes: Iterable[int] | Sized,
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
    validate_args(input_size, output_size, hidden_sizes, activation_functions)

    return nn.Sequential(*create_layers(input_size, output_size, hidden_sizes,
                                        activation_functions))


def get_params(tabular: dict | pd.DataFrame, keys: tuple) -> list:
    """
    Destructures a tabular input.
    :param keys: A tuple containing keys for specific use case.
    :param tabular: A dict or pd.DataFrame input.
    :return: A tuple containing values relevant to the `keys` list.
    """

    is_valid_keys(tabular, keys)

    return is_dict(tabular, keys) if isinstance(tabular, dict) else is_df(
        tabular, keys)


def is_valid_keys(tabular: dict | pd.DataFrame, keys: tuple) -> None:
    if not all(key in tabular for key in keys):
        raise ValueError(f"Tabular data must contain keys {keys}")


def is_dict(tabular: dict | pd.DataFrame, keys: tuple) -> list:
    params = []
    for key in keys:
        params.append(tabular[key])
    return params


def is_df(tabular: dict | pd.DataFrame, keys: tuple) -> list:
    params = []
    for key in keys:
        value = tabular[key].iloc[0]

        if isinstance(value, np.integer):
            params.append(value.item())
        else:
            params.append(value)
    return params
