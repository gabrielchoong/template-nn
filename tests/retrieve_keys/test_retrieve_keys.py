import pytest
import os
import json
from unittest.mock import mock_open
from template_nn.retrieve_keys import get_model_keys, KEY_FILE_PATH

# Mock data for keys.json
MOCK_KEYS_DATA = {
    "FNN": ["input_size", "output_size"],
    "CNN": ["image_size", "conv_channels"],
}


def test_get_model_keys_success(monkeypatch):
    # Mock os.path.exists to return True
    monkeypatch.setattr(os.path, "exists", lambda x: True)

    # Mock builtins.open to return a mock file object with JSON data
    monkeypatch.setattr(
        "builtins.open",
        lambda f, mode: mock_open(read_data=json.dumps(MOCK_KEYS_DATA))(),
    )

    keys = get_model_keys("FNN")
    assert keys == ["input_size", "output_size"]


def test_get_model_keys_file_not_found(monkeypatch):
    # Mock os.path.exists to return False
    monkeypatch.setattr(os.path, "exists", lambda x: False)

    with pytest.raises(
        FileNotFoundError, match=f"keys.json not found at {KEY_FILE_PATH}"
    ):
        get_model_keys("FNN")


def test_get_model_keys_model_not_found(monkeypatch):
    # Mock os.path.exists to return True
    monkeypatch.setattr(os.path, "exists", lambda x: True)

    # Mock builtins.open to return a mock file object with JSON data
    monkeypatch.setattr(
        "builtins.open",
        lambda f, mode: mock_open(read_data=json.dumps(MOCK_KEYS_DATA))(),
    )

    with pytest.raises(ValueError, match="Model name 'RNN' not found in keys.json"):
        get_model_keys("RNN")
