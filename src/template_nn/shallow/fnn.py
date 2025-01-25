from typing import Iterable, overload

import pandas as pd
import torch
import torch.nn as nn

from template_nn.utils.model_compose import build_dict_model, build_df_model, build_norm_model


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

    @overload
    def __init__(self, tabular: dict | pd.DataFrame | None = None, *args, **kwargs) -> None:
        ...

    @overload
    def __init__(self,
                 input_size: int | None = None,
                 output_size: int | None = None,
                 hidden_sizes: Iterable[int] | None = None,
                 activation_functions: list[callable] | None = None):
        ...

    def __init__(self,
                 input_size: int | None = None,
                 output_size: int | None = None,
                 hidden_sizes: Iterable[int] | None = None,
                 tabular: dict | pd.DataFrame | None = None,
                 activation_functions: list[callable] | None = None) -> None:

        """
        Initialises the neural network with parameters:
        :param input_size: int
        :param output_size: int
        :param hidden_sizes: Iterable[int]
        :param activation_functions: list[callable]
        """

        super(F_NN, self).__init__()

        if isinstance(tabular, dict):
            self.model = build_dict_model(tabular)

        if isinstance(tabular, pd.DataFrame):
            self.model = build_df_model(tabular)

        if tabular is None:
            self.model = build_norm_model(input_size, output_size, hidden_sizes, activation_functions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)
