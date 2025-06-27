import pandas as pd
import torch.nn as nn

from template_nn import FNN


def test_get_params_with_dict():
    tabular = {
        "input_size": 10,
        "output_size": 1,
        "hidden_sizes": [5],
        "activation_functions": [nn.ReLU()]
    }
    keys = ("input_size", "output_size", "hidden_sizes",
            "activation_functions")
    model = FNN(tabular)
    result = model._get_params(tabular, keys)
    assert result[:3] == [10, 1, [5]]
    assert isinstance(result[3][0], nn.ReLU)


def test_get_params_with_dataframe():
    df = pd.DataFrame([{
        "input_size": 10,
        "output_size": 1,
        "hidden_sizes": [5],
        "activation_functions": [nn.ReLU()]
    }])
    keys = ("input_size", "output_size", "hidden_sizes",
            "activation_functions")
    model = FNN(df)
    result = model._get_params(df, keys)
    assert result[:3] == [10, 1, [5]]
    assert isinstance(result[3][0], nn.ReLU)
