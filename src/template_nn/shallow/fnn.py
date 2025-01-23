import warnings
from typing import Iterable

import torch
import torch.nn as nn

from template_nn.utils.args_val import validate_args, iterable_to_list


class F_NN(nn.Module):
    """
    A Feedforward Neural Network (F_NN) model for supervised learning.

    The model learns the parameter \(\\beta\) based on input features \(X\) and corresponding output labels.

    Mathematical Formulation:
        - Hidden layer activation: \( H = f(WX + B) \)
        - Output layer prediction: \( y = H \\beta + \sigma \)

    The parameters learned during training are denoted by \(\\beta\), while \(\sigma\) represents the noise term (or error).

    The objective function for training is the Mean Squared Error (MSE) between the predicted output and actual labels:
        - \( J = \\arg\min(E) \)
        - \( E = \\text{MSE}(\\beta) \)

    References:
        - Suganthan, P. N., & Katuwal, R. (2021). On the origins of randomization-based feedforward neural networks.
          *Applied Soft Computing*, 105, 107239. [DOI: 10.1016/j.asoc.2021.107239](https://doi.org/10.1016/j.asoc.2021.107239)

    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_sizes: Iterable[int],
                 hidden_layer_num: int | None = None,
                 activation_functions: list[callable] = None) -> None:
        """
        Initialises the neural network with parameters:
        :param input_size: int
        :param output_size: int
        :param hidden_layer_num: int
        :param hidden_sizes: Iterable[int]
        :param activation_functions: list[callable]
        """

        super(F_NN, self).__init__()

        # missing arguments will result in errors that are hard to debug
        self.input_size, self.output_size, self.hidden_layer_num, self.hidden_sizes, self.activation_functions \
            = validate_args(input_size, output_size, hidden_layer_num, hidden_sizes, activation_functions)

        if self.hidden_layer_num >= 3:
            warnings.warn(
                "The network is considered deep (>=3 hidden layers). Consider using model templates from the 'deep' directory for better architecture options.",
                UserWarning
            )
        else:
            warnings.warn(
                "A shallow neural network (<=2 hidden layers) is being used. If you need more complexity, consider switching to a deeper architecture.",
                UserWarning
            )

        self.hidden_sizes = iterable_to_list(self.hidden_sizes)

        layers = []

        in_size = self.input_size

        # TODO: abstract layer generation logic
        for i, (_hidden_size, _activation_function) in enumerate(zip(self.hidden_sizes, self.activation_functions)):
            layers.append(nn.Linear(in_size, _hidden_size))
            layers.append(_activation_function)

            # sets in_size to the current hidden_size
            # effectively shifts the input size for the next layer
            in_size = _hidden_size

        layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)
