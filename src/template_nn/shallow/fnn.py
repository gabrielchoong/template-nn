from typing import Iterable, overload

import pandas as pd
import torch
import torch.nn as nn

from template_nn.utils.model_compose import build_norm_model, build_tabular_model


class F_NN(nn.Module):
    """
    A Feedforward Neural Network (F_NN) model for supervised learning.

    The model learns the parameter (beta) based on input features (X) and corresponding output labels.

    Mathematical Formulation:
        - Hidden layer activation: ( H = f(WX + B) )
        - Output layer prediction: ( y = H beta + sigma )

    The parameters learned during training are denoted by (beta), while (sigma) represents the noise term (or error).

    The objective function for training is the Mean Squared Error (MSE) between the predicted output and actual labels:
        - ( J = argmin(E) )
        - ( E = MSE(beta) )

    References:
        - Suganthan, P. N., & Katuwal, R. (2021). On the origins of randomization-based feedforward neural networks.
          *Applied Soft Computing*, 105, 107239. [DOI: 10.1016/j.asoc.2021.107239](https://doi.org/10.1016/j.asoc.2021.107239)

    """

    @overload
    def __init__(self, tabular: dict | pd.DataFrame | None = None,
                 visualise: bool = False) -> None:
        ...

    @overload
    def __init__(self,
                 input_size: int = None,
                 output_size: int = None,
                 hidden_sizes: Iterable[int] | int = None,
                 activation_functions: Iterable[nn.Module] | None = None,
                 visualise: bool = False):
        ...

    def __init__(self,
                 input_size: int = None,
                 output_size: int = None,
                 hidden_sizes: Iterable[int] | int = None,
                 tabular: dict | pd.DataFrame | None = None,
                 activation_functions: Iterable[nn.Module] | None = None,
                 visualise: bool = False) -> None:

        """
        The 3 required initializer arguments are `input_size`, `output_size`, and `hidden_sizes`.
        :param input_size: The number of input features for the model.
        :param output_size: The number of output features for the model.
        :param hidden_sizes: A collection of hidden layer node count of the model.
        :param tabular: An optional input accepting both a dictionary or a pandas.DataFrame object.
        :param activation_functions: An optional collection of activation functions for the model.
        :param visualise: A toggle switch to visualize the model. OFF(False) by default.
        """

        super(F_NN, self).__init__()

        if tabular is not None:
            self.model = build_tabular_model(tabular)
        else:
            self.model = build_norm_model(input_size, output_size, hidden_sizes, activation_functions)

        if visualise:
            print(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)
