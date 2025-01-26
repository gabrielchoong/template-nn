import warnings
from typing import Iterable, Sized

import torch.nn as nn


def validate_args(input_size: int,
                  output_size: int,
                  hidden_sizes: Iterable[int],
                  activation_functions: Iterable[nn.Module]
                  ) -> tuple[int, int, Sized, Iterable[nn.Module]]:
    """
    Validates user arguments.
    :param input_size: The number of input features.
    :param output_size: The number of output features.
    :param hidden_sizes: A collection of hidden node counts.
    :param activation_functions: An optional collection of activation functions.
    :return: A tuple containing (int, int, Sized, Iterable[nn.Module]).
    Unpack the values in the order of: input_size, output_size, hidden_sizes, activation_functions
    """

    # neural networks MUST have positive number of input features
    if not isinstance(input_size, int) or input_size <= 0:
        raise ValueError(f"Expected positive integer for input_size, got {input_size} instead.")

    # ... output features
    if not isinstance(output_size, int) or output_size <= 0:
        raise ValueError(f"Expected positive integer for output_size, got {output_size} instead.")

    # hidden sizes cannot be empty
    if not hidden_sizes:
        raise ValueError(f"Expected at least one hidden node count, got {hidden_sizes} instead.")

    # allowed single hidden layer neural network
    # to be instantiated with `hidden_sizes=5` (for example)
    # without having to wrap it: [5]
    if not isinstance(hidden_sizes, Iterable):
        hidden_sizes = [hidden_sizes]

    # 1. a non-integer value cannot be passed into model instantiation
    # 2. a node size of 0 is not defined for neural networks
    #    refrain user from passing integer values less than 0
    #
    # `map()` and `lambda` could be decomposed into the function below:
    # for x in hidden_sizes:
    #     if not isinstance(x, int) or x <= 0:
    #         return ValueError(...)
    if not all(map(lambda x: isinstance(x, int) and x > 0, hidden_sizes)):
        raise ValueError(f"Expected iterable of integers for hidden_sizes, got {hidden_sizes} instead.")

    # don't invert the order of these two checks
    # the error: `TypeError: object of type 'NoneType' has no len()` will occur
    # if the checks are inverted since `NoneType`object has no length
    if not activation_functions:
        activation_functions = [nn.ReLU()] * len(hidden_sizes)

    # reduces user errors by filling in the missing functions
    # assume same activation functions for users
    # included a warning to alert user
    if len(activation_functions) != len(hidden_sizes):
        warnings.warn("The number of activation functions provided doesn't match the number of hidden layers. "
                      "Using the last activation function for the remaining layers."
                      f"Expected {len(activation_functions)} activation functions, got {len(hidden_sizes)}."
                      f"Missing {len(hidden_sizes) - len(activation_functions)} activation functions.")

        # suppose 3 required, but only 2 were given
        # functions to be added = f(x) * (3-2)
        # generalised
        activation_functions += [activation_functions[-1]] * (len(hidden_sizes) - len(activation_functions))

    return input_size, output_size, hidden_sizes, activation_functions
