import pandas as pd
import pytest
import torch

from template_nn.shallow.fnn import F_NN


@pytest.mark.parametrize("input_size, expected_output_shape", [
    (10, (1, 2)),
    (20, (1, 2)),
    (50, (1, 2)),
])
def test_variable_input_size_shallow(input_size, expected_output_shape):
    # input_size - 5 - 2 Neural Network
    model = F_NN(input_size, output_size=2, hidden_sizes=[5])

    x = torch.randn(1, input_size)

    output = model(x)

    assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, but got {output.shape}"


@pytest.mark.parametrize("input_size, expected_output_shape", [
    (10, (1, 2)),
    (20, (1, 2)),
    (50, (1, 2)),
])
def test_variable_input_size_deep(input_size, expected_output_shape):
    # input_size - 5 - 2 Neural Network
    model = F_NN(input_size, output_size=2, hidden_sizes=[5 for _ in range(5)])

    x = torch.randn(1, input_size)

    output = model(x)

    assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, but got {output.shape}"


@pytest.mark.parametrize("input_size, expected_output_shape", [
    (10, (1, 2)),
    (20, (1, 2)),
    (50, (1, 2)),
])
def test_none_type_hidden_layer_num(input_size, expected_output_shape):
    # input_size - 5 - 2 Neural Network
    model = F_NN(input_size=input_size, output_size=2, hidden_sizes=[5])

    x = torch.randn(1, input_size)

    output = model(x)

    assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, but got {output.shape}"


# Testing build_dict_model
def test_build_dict_model_valid():
    valid_dict = {
        "input_size": 10,
        "output_size": 2,
        "hidden_sizes": [5],
        "activation_functions": [torch.nn.ReLU()]
    }

    model = F_NN(tabular=valid_dict)
    x = torch.randn(1, valid_dict["input_size"])
    output = model(x)

    assert output.shape == (
        1, valid_dict["output_size"]), f"Expected output shape {(1, valid_dict['output_size'])}, but got {output.shape}"


def test_build_dict_model_invalid_keys():
    invalid_dict = {
        "input_size": 10,
        "output_size": 2,
        "hidden_sizes": [5]
        # Missing "activation_functions" key
    }

    with pytest.raises(ValueError, match="Tabular data must contain keys"):
        F_NN(tabular=invalid_dict)


# Testing build_df_model
def test_build_df_model_valid():
    valid_df = pd.DataFrame({
        "input_size": [10],
        "output_size": [2],
        "hidden_sizes": [[5]],
        "activation_functions": [[torch.nn.ReLU()]]
    })

    model = F_NN(tabular=valid_df)
    x = torch.randn(1, valid_df["input_size"].item())
    output = model(x)

    assert output.shape == (1, valid_df["output_size"].item()), \
        f"Expected output shape {(1, valid_df['output_size'].item())}, but got {output.shape}"


def test_build_df_model_invalid_columns():
    invalid_df = pd.DataFrame({
        "input_size": [10],
        "output_size": [2],
        "hidden_sizes": [[5]]
        # Missing "activation_functions" column
    })

    with pytest.raises(ValueError, match="Tabular data must contain keys"):
        F_NN(tabular=invalid_df)
