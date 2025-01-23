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
    model = F_NN(input_size, output_size=2, hidden_layer_num=1, hidden_sizes=[5])

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
    model = F_NN(input_size, output_size=2, hidden_layer_num=5, hidden_sizes=[5 for _ in range (5)])

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