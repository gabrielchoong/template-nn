from typing import List, Iterable, Sized

from torch import nn


def create_layers(
        input_size: int, output_size: int, hidden_sizes: List[int] | Sized,
        activation_functions: Iterable[nn.Module]) -> list[nn.Module]:
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

    for i, (hidden_size, activation_function) in enumerate(
            zip(hidden_sizes, activation_functions)):
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(activation_function)

        # sets in_size to the current hidden_size
        # effectively shifts the input size for the next layer
        in_size = hidden_size

    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    return layers
