import warnings
from typing import Iterable, List

import torch.nn as nn


def validate_args(input_size, output_size, hidden_layer_num, hidden_sizes, activation_functions):
    if not isinstance(input_size, int) or input_size <= 0:
        raise ValueError(f"Expected positive integer for input_size, got {input_size} instead.")

    if not isinstance(output_size, int) or output_size <= 0:
        raise ValueError(f"Expected positive integer for output_size, got {output_size} instead.")

    if hidden_layer_num is None:
        hidden_layer_num = len(hidden_sizes)

    if len(hidden_sizes) != hidden_layer_num:
        raise ValueError(f"Mismatch between hidden_layer_num and hidden_size length: "
                         f"expected {hidden_layer_num}, but got {len(hidden_sizes)}.")

    if activation_functions is None:
        activation_functions = [nn.ReLU()] * hidden_layer_num

    if len(activation_functions) != hidden_layer_num:
        warnings.warn("The number of activation functions provided doesn't match the number of hidden layers. "
                      "Using the last activation function for the remaining layers.")

        # suppose 3 required, but only 2 were given
        # functions to be added = f(x) * (3-2)
        # generalised
        activation_functions += [activation_functions[-1]] * (hidden_layer_num - len(activation_functions))

    return input_size, output_size, hidden_layer_num, hidden_sizes, activation_functions


def iterable_to_list(iterable: any) -> List:
    if isinstance(iterable, list):
        return iterable

    if isinstance(iterable, Iterable):
        try:
            return [int(item) for item in iterable]
        except ValueError:
            raise TypeError("All items in the iterable must be integers.")

    raise TypeError("Expected a list or iterable of integers.")
