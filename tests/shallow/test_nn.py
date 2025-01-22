import pytest
import torch
from template_nn.shallow.fnn import F_NN

@pytest.mark.parametrize("input_size, expected_output_shape", [
    (10, (1, 2)),
    (20, (1, 2)),
    (50, (1, 2)),
])
def test_variable_input_size(input_size, expected_output_shape):

    # input_size - 5 - 2 Neural Network
    model = F_NN(input_size=input_size, hidden_size=5, output_size=2)

    x = torch.randn(1, input_size)

    output = model(x)

    assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, but got {output.shape}"