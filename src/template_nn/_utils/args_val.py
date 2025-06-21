from typing import Iterable, Sized

import torch.nn as nn
from torch.nn import Module


def validate_args(input_size: int,
                  output_size: int,
                  hidden_sizes: Iterable[int] | Sized,
                  activation_functions: Iterable[nn.Module]
                  ) -> None:
    """
    Validates user arguments.
    :param input_size: The number of input features.
    :param output_size: The number of output features.
    :param hidden_sizes: A collection of hidden node counts.
    :param activation_functions: An optional collection of activation functions.
    :return: A tuple containing (int, int, Sized, Iterable[nn.Module]).
    Unpack the values in the order of: input_size, output_size, hidden_sizes, activation_functions
    """

    is_positive_int(input_size)
    is_positive_int(output_size)
    is_positive_iterable_int(hidden_sizes)

    has_activation_functions(activation_functions)
    activation_functions_check(activation_functions, hidden_sizes)


def is_positive_int(number: int) -> None:
    if not isinstance(number, int) or number <= 0:
        raise ValueError(f"Expected positive integer, got {number} instead.")

def is_positive_iterable_int(numbers: Iterable[int]) -> None:
    for num in numbers:
        if not isinstance(num, int) or num <= 0:
            raise ValueError(f"Expected positive integer, got {num} instead.")

def has_activation_functions(activation_functions: Iterable[Module]) -> None:
    if activation_functions is None:
        raise ValueError("No activation functions were provided.")

def activation_functions_check(activation_functions: Iterable[Module] | Sized, hidden_sizes: Iterable[int] | Sized) -> None:
    if len(activation_functions) != len(hidden_sizes):
        raise ValueError("Number of activation functions does not match number of hidden node counts.")