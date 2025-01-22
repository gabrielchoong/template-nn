import warnings

import torch
import torch.nn as nn


class F_NN(nn.Module):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_layer_num: int,
                 hidden_sizes: list[int] | torch.Tensor) -> None:

        super(F_NN, self).__init__()

        if len(hidden_sizes) != hidden_layer_num:
            raise ValueError(f"Mismatch between hidden_layer_num and hidden_size length: "
                             f"expected {hidden_layer_num}, but got {len(hidden_sizes)}.")

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

        if isinstance(hidden_sizes, torch.Tensor):
            hidden_sizes = list(hidden_sizes)

        layers = []

        in_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())

            # sets in_size to the current hidden_size
            # effectively shifts the input size for the next layer
            in_size = hidden_size

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)
