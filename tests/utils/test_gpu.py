import pytest
from unittest.mock import patch
from template_nn.utils.gpu import get_gpu_device


@pytest.mark.parametrize("cuda_available, mps_built, expected_device", [
    (True, False, "cuda"),  # when CUDA is available
    (False, True, "mps"),  # when MPS is available
    (True, True, "cuda"), # when both CUDA and MPS are available
    (False, False, "cpu"),  # when neither CUDA nor MPS is available
])
def test_get_gpu_device(cuda_available, mps_built, expected_device):
    with patch("torch.cuda.is_available", return_value=cuda_available), \
            patch("torch.backends.mps.is_built", return_value=mps_built), \
            patch("torch.cuda.get_device_name", return_value="Mock GPU" if cuda_available else ""):
        device = get_gpu_device()
        assert str(device) == expected_device