import warnings

import pandas as pd
import torch.nn as nn

from template_nn.utils.args_val import validate_args, iterable_to_list


def warn_hidden_layer(hidden_layer_num: int) -> None:
    if hidden_layer_num >= 3:
        warnings.warn(
            "The network is considered deep (>=3 hidden layers). Consider using model templates from the 'deep' directory for better architecture options.",
            UserWarning
        )
    else:
        warnings.warn(
            "A shallow neural network (<=2 hidden layers) is being used. If you need more complexity, consider switching to a deeper architecture.",
            UserWarning
        )


def create_layers(input_size,
                  hidden_sizes,
                  output_size,
                  activation_functions) -> list[nn.Module]:
    layers = []

    in_size = input_size

    # TODO: abstract layer generation logic
    for i, (hidden_size, activation_function) in enumerate(zip(hidden_sizes, activation_functions)):
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(activation_function)

        # sets in_size to the current hidden_size
        # effectively shifts the input size for the next layer
        in_size = hidden_size

    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    return layers


def get_params(tabular: dict | pd.DataFrame) -> tuple:
    if isinstance(tabular, dict):
        input_size, output_size = tabular["input_size"], tabular["output_size"]

        hidden_sizes = tabular["hidden_sizes"]

        activation_functions = tabular["activation_functions"]

        return input_size, output_size, hidden_sizes, activation_functions

    if isinstance(tabular, pd.DataFrame):
        input_size, output_size = tabular["input_size"].iloc[0].item(), tabular["output_size"].iloc[0].item()

        hidden_sizes = tabular["hidden_sizes"].iloc[0]

        activation_functions = tabular["activation_functions"].iloc[0]

        return input_size, output_size, hidden_sizes, activation_functions


def build_model(input_size,
                output_size,
                hidden_sizes,
                activation_functions) -> nn.Sequential:
    # missing arguments will result in errors that are hard to debug
    input_size, output_size, hidden_sizes, activation_functions \
        = validate_args(input_size, output_size, hidden_sizes, activation_functions)

    warn_hidden_layer(len(hidden_sizes))

    hidden_sizes = iterable_to_list(hidden_sizes)

    layers = create_layers(input_size, hidden_sizes, output_size, activation_functions)

    model = nn.Sequential(*layers)

    return model


keys = ["input_size", "output_size", "hidden_sizes", "activation_functions"]


def build_dict_model(dictionary: dict) -> nn.Sequential:
    """
    :param dictionary: dict
    :return: torch.nn.Sequential
    """

    if not all(key in dictionary for key in keys):
        raise ValueError(f"Dictionary must contain keys {keys}")

    input_size, output_size, hidden_sizes, activation_functions = get_params(dictionary)

    return build_model(input_size, output_size, hidden_sizes, activation_functions)


def build_df_model(dataFrame: pd.DataFrame) -> nn.Sequential:
    """
    :param dataFrame: pd.DataFrame
    :return: torch.nn.Sequential
    """

    if not all(key in dataFrame.columns for key in keys):
        raise ValueError(f"DataFrame must contain keys {keys}")

    input_size, output_size, hidden_sizes, activation_functions = get_params(dataFrame)

    return build_model(input_size, output_size, hidden_sizes, activation_functions)


def build_norm_model(input_size,
                     output_size,
                     hidden_sizes,
                     activation_functions) -> nn.Sequential:
    """
    :param input_size: int
    :param output_size: int
    :param hidden_sizes: Iterable[int]
    :param activation_functions: list[callable]
    :return: nn.Sequential
    """
    return build_model(input_size, output_size, hidden_sizes, activation_functions)
