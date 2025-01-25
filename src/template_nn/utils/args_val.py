import warnings
from typing import Iterable, List

import torch.nn as nn


def validate_args(input_size, output_size, hidden_sizes, activation_functions):
    if not isinstance(input_size, int) or input_size <= 0:
        raise ValueError(f"Expected positive integer for input_size, got {input_size} instead.")

    if not isinstance(output_size, int) or output_size <= 0:
        raise ValueError(f"Expected positive integer for output_size, got {output_size} instead.")

    if activation_functions is None:
        activation_functions = [nn.ReLU()] * len(hidden_sizes)

    if len(activation_functions) != len(hidden_sizes):
        warnings.warn("The number of activation functions provided doesn't match the number of hidden layers. "
                      "Using the last activation function for the remaining layers.")

        # suppose 3 required, but only 2 were given
        # functions to be added = f(x) * (3-2)
        # generalised
        activation_functions += [activation_functions[-1]] * (len(hidden_sizes) - len(activation_functions))

    return input_size, output_size, hidden_sizes, activation_functions


def iterable_to_list(iterable: any) -> List:
    if isinstance(iterable, list):
        return iterable

    if isinstance(iterable, Iterable):
        try:
            return [int(item) for item in iterable]
        except ValueError:
            raise TypeError("All items in the iterable must be integers.")

    raise TypeError("Expected a list or iterable of integers.")
